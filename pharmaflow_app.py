import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import io
from datetime import datetime

# Statistical modeling
import pymc3 as pm
import arviz as az
from lifelines import CoxPHFitter, WeibullFitter
from sklearn.ensemble import GradientBoostingRegressor

# PDF export
from fpdf import FPDF

# ETL orchestration
from prefect import task, Flow
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & CACHE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaFlow 2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache external API calls
def cache_data(ttl_seconds=3600):
    def decorator(func):
        return st.cache_data(ttl=ttl_seconds)(func)
    return decorator

@cache_data(86400)
def fetch_openfda_shortages(limit=1000):
    url = f"https://api.fda.gov/drug/shortages.json?limit={limit}"
    r = requests.get(url)
    data = r.json().get('results', [])
    return pd.json_normalize(data, sep='.')

@cache_data(86400)
def fetch_nadac():
    url = (
        "https://data.medicaid.gov/api/1/datastore/query/"
        "4d7af295-2132-55a8-b40c-d6630061f3e8/0/download?format=csv"
    )
    r = requests.get(url)
    return pd.read_csv(io.StringIO(r.text))

@task(max_retries=3)
@cache_data(86400)
def fetch_orange_book():
    url = 'https://www.accessdata.fda.gov/scripts/cder/ob/docs/obpxml.cfm'
    r = requests.get(url, timeout=30)
    root = ET.fromstring(r.content)
    records = []
    for prod in root.findall('.//product'):
        records.append({
            'rxcui': prod.findtext('rxcui'),
            'trade_name': prod.findtext('trade_name'),
            'patent_exclusivities': [p.text for p in prod.findall('patentExclusivity')]
        })
    return pd.DataFrame(records)

@task(max_retries=3)
@cache_data(86400)
def fetch_generic_history():
    # Placeholder: load a curated CSV of historical generic entries
    return pd.read_csv('generic_entry_history.csv')

# -----------------------------------------------------------------------------
# 2. ETL PIPELINE
# -----------------------------------------------------------------------------
with Flow("ingest_all", result=LocalResult(dir="./data", serializer=PandasSerializer())) as flow:
    shortages = fetch_openfda_shortages()
    nadac = fetch_nadac()
    orange = fetch_orange_book()
    history = fetch_generic_history()
# flow.run()  # Uncomment to run ETL on demand

# -----------------------------------------------------------------------------
# 3. DATA NORMALIZATION & MDM
# -----------------------------------------------------------------------------
def normalize_ids(df, col, length=10):
    df[col] = df[col].astype(str).str.zfill(length)
    return df

# -----------------------------------------------------------------------------
# 4. MODELING COMPONENTS
# -----------------------------------------------------------------------------

def fit_survival_model(history_df):
    """
    Fit a Cox Proportional Hazards model on time-to-generic data.
    history_df must contain: 'time_to_entry', 'event_observed', covariates...
    """
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(history_df, duration_col='time_to_entry', event_col='event_observed')
    return cph


def fit_price_drop_model(history_df):
    """
    Fit hierarchical Bayesian model for price-drop rates.
    history_df: columns ['drug_class', 'n_generics', 'price_ratio']
    """
    drug_classes = history_df['drug_class'].astype('category')
    history_df['class_idx'] = drug_classes.cat.codes
    n_classes = len(drug_classes.cat.categories)

    with pm.Model() as model:
        mu_a = pm.Normal('mu_a', 0.5, 0.2)
        sigma_a = pm.HalfNormal('sigma_a', 0.1)
        a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_classes)

        mu_b = pm.Normal('mu_b', -0.1, 0.1)
        sigma_b = pm.HalfNormal('sigma_b', 0.05)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_classes)

        obs = pm.Normal(
            'obs',
            mu=a[history_df['class_idx'].values] + b[history_df['class_idx'].values] * history_df['n_generics'].values,
            sigma=0.05,
            observed=history_df['price_ratio'].values
        )
        trace = pm.sample(1000, tune=1000, target_accept=0.92, cores=2)
    return model, trace, drug_classes

# Forecast helper combining survival & price models

def simulate_price_trajectory(drug_id, exclusivity, n_generics_dist, horizon, n_sim=1000, cph=None, bayes=None):
    """
    Simulate price ratios over time for a single drug.
    """
    time_grid = np.linspace(0, horizon, 50)
    sims = []
    for _ in range(n_sim):
        # sample time to entry
        if cph:
            # use median covariate values for this drug
            t_entry = cph.predict_median(pd.DataFrame([cph.baseline_cumulative_hazard_]))
            t_entry = t_entry.values[0]
        else:
            t_entry = np.random.normal(exclusivity, exclusivity * 0.2)
        # sample drop rate
        if bayes:
            # draw random a,b for drug's class
            class_idx = bayes[2].cat.categories.get_loc(drug_id)
            a_samp = np.random.choice(bayes[1].posterior['a'].stack(draws=('chain','draw')).values[:,class_idx])
            b_samp = np.random.choice(bayes[1].posterior['b'].stack(draws=('chain','draw')).values[:,class_idx])
            drop_rate = min(max(1 - (a_samp + b_samp * np.mean(n_generics_dist)), 0),1)
        else:
            drop_rate = np.random.beta(2, 5)
        # build trajectory
        ratios = [1.0 if t<t_entry else (1-drop_rate) for t in time_grid]
        sims.append(ratios)
    return time_grid, np.array(sims)

# -----------------------------------------------------------------------------
# 5. STREAMLIT UI PAGES
# -----------------------------------------------------------------------------

def main():
    st.title("PharmaFlow 2.0: Policy Simulation & Analytics")
    pages = {
        "Dashboard": page_dashboard,
        "Drug Drilldown": page_drug,
        "Class Comparison": page_class,
        "Scenario Builder": page_scenario,
        "Export Report": page_export,
        "Methodology": page_methodology
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[choice]()

# --- Dashboard ---
def page_dashboard():
    st.header("📈 National Overview")
    shortages = fetch_openfda_shortages()
    nadac = fetch_nadac()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Drug Shortages")
        st.dataframe(
            shortages[['product.product_ndc','product.trade_name','report_date']]
            .rename(columns={'product.product_ndc':'NDC', 'product.trade_name':'Drug', 'report_date':'Date'})
            .sort_values('Date', ascending=False).head(10)
        )
    with col2:
        st.subheader("Average NADAC Price Trajectory")
        nadac['As of Date'] = pd.to_datetime(nadac['As of Date'])
        price_ts = nadac.groupby('As of Date')['NADAC Per Unit'].mean()
        st.line_chart(price_ts.rename_axis('Date').to_frame('Avg NADAC'))

# --- Drug Drilldown ---

def page_drug():
    st.header("🔍 Drug-level Drilldown & Forecast")
    ndc = st.text_input("Enter NDC Code (10-digit)")
    if ndc:
        # Normalize and fetch historical data
        history = fetch_generic_history()
        history = normalize_ids(history, 'ndc', 10)
        drug_hist = history[history['ndc']==ndc]
        if drug_hist.empty:
            st.warning("No historical data found for this NDC.")
            return

        # Survival fit
        cph = fit_survival_model(drug_hist)
        st.subheader("Time-to-Generic Survival Function")
        surv_df = cph.predict_survival_function(drug_hist.iloc[[0]])
        st.line_chart(surv_df)

        # Price-drop model
        bayes_model, bayes_trace, classes = fit_price_drop_model(history)

        # Simulation
        st.subheader("Price Ratio Simulations")
        exclusivity = st.slider("Override Exclusivity (years)",1,20,5)
        dist = np.random.poisson(5, size=100)  # placeholder distribution
        times, sims = simulate_price_trajectory(ndc, exclusivity, dist, horizon=10,
                                                n_sim=500, cph=cph, bayes=(bayes_model,bayes_trace,classes))
        sim_df = pd.DataFrame(sims.T, index=times)
        st.line_chart(sim_df)
        avg_final = 1 - sim_df.iloc[-1].mean()
        ci = np.percentile(1-sim_df.iloc[-1], [2.5,97.5])
        st.markdown(f"**Projected Drop at 10y:** {avg_final:.0%} (95% CI {ci[0]:.0%}-{ci[1]:.0%})")

# --- Class Comparison ---
def page_class():
    st.header("💊 Therapeutic Class Comparison")
    history = fetch_generic_history()
    model, trace, classes = fit_price_drop_model(history)

    import matplotlib.pyplot as plt
    import seaborn as sns  # For nicer violin plots
    fig, ax = plt.subplots(figsize=(10,5))
    # Extract posterior drop rates per class
    drops = []
    for i, cls in enumerate(classes.cat.categories):
        a_samps = trace.posterior['a'].stack(draws=('chain','draw')).values[:,i]
        b_samps = trace.posterior['b'].stack(draws=('chain','draw')).values[:,i]
        drops.append(1 - (a_samps + b_samps*5))
    sns.violinplot(data=drops, ax=ax)
    ax.set_xticklabels(classes.cat.categories)
    ax.set_ylabel('Price Drop Rate')
    st.pyplot(fig)

# --- Scenario Builder ---
def page_scenario():
    st.header("⚙️ Policy Scenario Builder")
    exclusivity = st.number_input("Exclusivity (years)",1,20,5)
    competitors = st.number_input("Expected # Generics",1,20,10)
    horizon = st.number_input("Time Horizon (years)",1,20,10)
    sims = st.number_input("MC Simulations",100,5000,1000)

    # Run simple parametric sim if models not loaded
    times = np.linspace(0, horizon, 50)
    sims_arr = np.zeros((sims,len(times)))
    for i in range(sims):
        t_entry = np.random.normal(exclusivity, exclusivity*0.2)
        drop = min(max(np.random.normal(competitors/competitors,0.1),0),1)
        sims_arr[i] = [1.0 if t<t_entry else 1-drop for t in times]
    st.subheader("Simulation Overview")
    sim_df = pd.DataFrame(sims_arr.T, index=times)
    st.line_chart(sim_df)

# --- Export Report ---
def page_export():
    st.header("📄 Export Policy Brief")
    st.markdown("Generate a PDF with the current dashboard views and key metrics.")
    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200,10,txt="PharmaFlow 2.0 Policy Brief",ln=1,align='C')
        pdf.cell(0,10,txt=f"Generated: {datetime.now().strftime('%Y-%m-%d')}",ln=1)
        # Additional content embedding charts would require saving images and adding via pdf.image()
        out = io.BytesIO()
        pdf.output(out)
        st.download_button("Download PDF", data=out.getvalue(), file_name="pharmaflow_report.pdf")

# --- Methodology ---
def page_methodology():
    st.header("🔬 Methodology & Assumptions")
    st.markdown("**Data Sources**:")
    st.markdown("- OpenFDA Drug Shortages API")
    st.markdown("- Medicaid NADAC pricing")
    st.markdown("- FDA Orange Book XML")
    st.markdown("- Historical generic entry dataset")
    st.markdown("\n**Statistical Models**:")
    st.markdown("- Cox Proportional Hazards for generic entry timing")
    st.markdown("- Hierarchical Bayesian pricing model with PyMC3")
    st.markdown("- Monte Carlo simulation for scenario analysis")
    st.markdown("\n**Future Extensions**:")
    st.markdown("- Supply‐side risk via system‐dynamics")
    st.markdown("- Agent‐based firm entry/exit modeling")
    st.markdown("- Claims‐level out‐of‐pocket impact estimation")

if __name__ == '__main__':
    main()
