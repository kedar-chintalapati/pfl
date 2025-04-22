import streamlit as st
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import to_long_format, add_covariate_to_timeline
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaFlow 2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DATA FETCHING FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_openfda_shortages(limit=1000):
    """Fetch live drug shortage data from openFDA."""
    url = f"https://api.fda.gov/drug/shortages.json?limit={limit}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json().get('results', [])
    df = pd.json_normalize(data, sep='.')
    # standardize date
    if 'report_date' in df.columns:
        df['report_date'] = pd.to_datetime(df['report_date'])
    return df

@st.cache_data(ttl=3600)
def fetch_nadac():
    """Fetch Medicaid NADAC pricing data."""
    url = (
        "https://data.medicaid.gov/api/1/datastore/query/"
        "4d7af295-2132-55a8-b40c-d6630061f3e8/0/download?format=csv"
    )
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df['As of Date'] = pd.to_datetime(df['As of Date'])
    return df

@st.cache_data(ttl=86400)
def fetch_orange_book():
    """Fetch patent exclusivity data from FDA Orange Book XML."""
    url = 'https://www.accessdata.fda.gov/scripts/cder/ob/docs/obpxml.cfm'
    r = requests.get(url)
    r.raise_for_status()
    # parse XML
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r.content)
    records = []
    for prod in root.findall('.//product'):
        rxcui = prod.findtext('rxcui')
        exclusivities = [ex.text for ex in prod.findall('patentExclusivity')]
        records.append({'rxcui': rxcui, 'exclusivities': exclusivities})
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 3. DATA PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def preprocess_data(shortages, nadac, orange):
    """Merge and normalize data sources into a master drug table."""
    # Extract shortages summary: count by NDC
    shortages['product.product_ndc'] = shortages['product.product_ndc'].astype(str)
    shortage_counts = (
        shortages.groupby('product.product_ndc')
                .size()
                .reset_index(name='shortage_events')
    )
    # NADAC: latest price per NDC
    nadac_latest = (
        nadac.sort_values('As of Date')
              .groupby('NDC')
              .tail(1)[['NDC', 'NADAC Per Unit']]
              .rename(columns={'NADAC Per Unit': 'nadac_price'})
    )
    # Orange Book: simplify exclusivity to max years
    orange['max_exclusivity_yrs'] = (
        orange['exclusivities']
              .apply(lambda lst: max([int(x) for x in lst]) if lst else np.nan)
    )
    # Merge all
    df = shortage_counts.merge(
        nadac_latest, left_on='product.product_ndc', right_on='NDC', how='outer'
    )
    df = df.merge(
        orange[['rxcui', 'max_exclusivity_yrs']],
        left_on='product.product_ndc', right_on='rxcui', how='left'
    )
    # Fill missing
    df['shortage_events'] = df['shortage_events'].fillna(0)
    df['nadac_price'] = df['nadac_price'].fillna(df['nadac_price'].mean())
    df['max_exclusivity_yrs'] = df['max_exclusivity_yrs'].fillna(0)
    # Derive drug class from NDC prefix: placeholder mapping
    df['drug_class'] = df['product.product_ndc'].str[:4]  # simplistic
    return df

# -----------------------------------------------------------------------------
# 4. MODEL FITTING
# -----------------------------------------------------------------------------
@st.cache_data
def fit_survival_model(df):
    """Fit a CoxPH model for time-to-first-generic entry."""
    # For demonstration, generate synthetic survival data
    # duration: time to generic entry (yrs), event: 1 if generic entered
    n = len(df)
    np.random.seed(0)
    df_surv = pd.DataFrame({
        'duration': np.random.exponential(scale=5, size=n),
        'event': (np.random.rand(n) < 0.7).astype(int),
        'shortage_events': df['shortage_events'],
        'max_exclusivity_yrs': df['max_exclusivity_yrs']
    })
    cph = CoxPHFitter()
    cph.fit(df_surv, duration_col='duration', event_col='event', show_progress=False)
    return cph, df_surv

@st.cache_data
def fit_price_model(df):
    """Fit a BayesianRidge model for price drop vs number of generics and class."""
    # For demo, synthetic data: price_ratio ~ intercept + slope * n_generics + noise
    n = len(df)
    np.random.seed(1)
    df_model = pd.DataFrame({
        'price_ratio': np.clip(1 - np.random.beta(2, 5, size=n), 0.05, 1.0),
        'n_generics': np.random.poisson(lam=3, size=n),
        'drug_class': df['drug_class']
    })
    # One-hot encode drug_class
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_class = encoder.fit_transform(df_model[['drug_class']])
    X = np.hstack([X_class, df_model[['n_generics']].values])
    y = df_model['price_ratio'].values
    model = BayesianRidge()
    model.fit(X, y)
    return model, encoder

# -----------------------------------------------------------------------------
# 5. STREAMLIT PAGES
# -----------------------------------------------------------------------------

def main():
    st.title("PharmaFlow 2.0: Policy Simulation & Analytics")
    pages = ["National Overview", "Drug Drilldown", "Class Comparison", "Scenario Builder", "Methodology"]
    choice = st.sidebar.selectbox("Select page", pages)
    # Load and preprocess data
    shortages = fetch_openfda_shortages()
    nadac = fetch_nadac()
    orange = fetch_orange_book()
    master = preprocess_data(shortages, nadac, orange)
    # Fit models
    cph_model, surv_df = fit_survival_model(master)
    price_model, class_encoder = fit_price_model(master)
    # Page routing
    if choice == "National Overview":
        page_overview(master, nadac)
    elif choice == "Drug Drilldown":
        page_drug(master, cph_model, price_model, class_encoder)
    elif choice == "Class Comparison":
        page_class(master, cph_model, price_model, class_encoder)
    elif choice == "Scenario Builder":
        page_scenario(master, cph_model, price_model, class_encoder)
    else:
        page_methodology()

# --- Page: Overview ---
def page_overview(master, nadac):
    st.header("National Overview")
    # Shortage summary
    st.subheader("Drug Shortages")
    st.metric("Total drugs with shortages", int((master['shortage_events']>0).sum()))
    # NADAC trend
    st.subheader("Average Medicaid NADAC Price Over Time")
    avg_price = nadac.groupby('As of Date')['NADAC Per Unit'].mean().reset_index()
    st.line_chart(avg_price.rename(columns={'As of Date': 'index', 'NADAC Per Unit': 'Avg Price'}).set_index('index'))

# --- Page: Drug Drilldown ---
def page_drug(master, cph_model, price_model, class_encoder):
    st.header("Drug-level Drilldown & Forecast")
    ndc_list = master['product.product_ndc'].unique().tolist()
    selected = st.selectbox("Select NDC", ndc_list)
    drug_df = master[master['product.product_ndc']==selected]
    if drug_df.empty:
        st.write("No data for selected drug.")
        return
    # Display basic info
    st.write("**NDC:**", selected)
    st.write("**Current NADAC price:** $", float(drug_df['nadac_price']))
    st.write("**Historical shortage events:**", int(drug_df['shortage_events']))
    st.write("**Patent exclusivity (yrs):**", float(drug_df['max_exclusivity_yrs']))
    # Survival forecast: time to generic entry
    kmf = KaplanMeierFitter()
    kmf.fit(durations=surv_df['duration'], event_observed=surv_df['event'])
    st.subheader("Kaplan-Meier: Time to Generic Entry")
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    st.pyplot(fig)
    # Price drop forecast via BayesianRidge
    st.subheader("Predicted Price Drop vs # Generics")
    n_range = np.arange(0, 11)
    # prepare features
    cls = drug_df['drug_class'].iloc[0]
    cls_enc = class_encoder.transform([[cls]])
    preds = []
    for n in n_range:
        X = np.hstack([cls_enc, [[n]]])
        pred = price_model.predict(X)[0]
        preds.append(pred)
    df_pred = pd.DataFrame({'n_generics': n_range, 'price_ratio': preds})
    st.line_chart(df_pred.set_index('n_generics')['price_ratio'])

# --- Page: Class Comparison ---
def page_class(master, cph_model, price_model, class_encoder):
    st.header("Therapeutic Class Comparison")
    classes = master['drug_class'].unique().tolist()
    selected = st.multiselect("Select classes", classes, default=classes[:3])
    if not selected:
        st.write("Choose at least one class.")
        return
    # Price drop distributions by class
    st.subheader("Price Ratio Distribution by # Generics")
    fig, ax = plt.subplots()
    for cls in selected:
        cls_enc = class_encoder.transform([[cls]])
        distributions = []
        for n in range(1, 6):
            X = np.hstack([cls_enc, [[n]]])
            # sample coef posterior ~ normal
            coef_mean = price_model.coef_
            coef_std = np.sqrt(np.diag(price_model.sigma_))
            sims = np.random.normal(coef_mean.dot(X.T), coef_std.dot(X.T), size=200)
            distributions.append((n, sims))
        # plot median
        medians = [np.median(s) for _, s in distributions]
        ax.plot(range(1,6), medians, label=cls)
    ax.set_xlabel("# Generics")
    ax.set_ylabel("Median Price Ratio")
    ax.legend()
    st.pyplot(fig)
    # Survival curves by class
    st.subheader("Time to Entry by Class (KM)")
    fig2, ax2 = plt.subplots()
    for cls in selected:
        mask = surv_df['shortage_events']>=0  # placeholder
        kmf = KaplanMeierFitter()
        kmf.fit(durations=surv_df['duration'], event_observed=surv_df['event'], label=cls)
        kmf.plot_survival_function(ax=ax2)
    st.pyplot(fig2)

# --- Page: Scenario Builder ---
def page_scenario(master, cph_model, price_model, class_encoder):
    st.header("Policy Scenario Builder")
    exclusivity = st.slider("Exclusivity (yrs)", 1, 20, 5)
    expected_generics = st.slider("Expected # Generics", 1, 15, 10)
    horizon = st.slider("Time Horizon (yrs)", 1, 20, 10)
    sims = st.number_input("Simulations", 100, 5000, 1000, step=100)

    # Monte Carlo simulation
    st.subheader("Simulated Price Trajectories")
    time_pts = np.linspace(0, horizon, 50)
    price_matrix = np.zeros((sims, len(time_pts)))
    # get average class encoding
    cls_list = master['drug_class'].unique().tolist()
    cls_encs = class_encoder.transform([[cls_list[0]]])
    for i in range(sims):
        # sample entry time from survival function
        t = cph_model.predict_survival_function(pd.DataFrame([
            {'shortage_events': master['shortage_events'].mean(),
             'max_exclusivity_yrs': exclusivity}
        ]), times=time_pts).values.flatten()
        # sample price drop
        drop_rate = 1 - price_model.predict(np.hstack([cls_encs, [[expected_generics]]]))[0]
        price_matrix[i, :] = np.where(time_pts < exclusivity, 1.0, 1-drop_rate)
    mean_ratio = price_matrix.mean(axis=0)
    df_chart = pd.DataFrame({'Avg Price Ratio': mean_ratio}, index=time_pts)
    st.line_chart(df_chart)

    # Summary at horizon
    final = price_matrix[:, -1]
    mean_drop = 1 - final.mean()
    lower, upper = np.percentile(final, [2.5, 97.5])
    ci_lo = 1 - upper
    ci_hi = 1 - lower
    st.markdown(f"**Projected price drop at {horizon} years:** {mean_drop*100:.1f}% (95% CI: {ci_lo*100:.1f}%–{ci_hi*100:.1f}%)")

# --- Page: Methodology ---
def page_methodology():
    st.header("Methodology & Assumptions")
    st.markdown("**Data Sources**")
    st.markdown("- openFDA Drug Shortages API")
    st.markdown("- Medicaid NADAC pricing CSV")
    st.markdown("- FDA Orange Book XML")
    st.markdown("**Statistical Models**")
    st.markdown("- Cox Proportional Hazards for time-to-entry (lifelines)")
    st.markdown("- Bayesian Ridge regression for price-drop estimation (scikit-learn)")
    st.markdown("**Future Extensions**")
    st.markdown("- Incorporate global market comparisons (EU, JP, CA)")
    st.markdown("- Patient-level out-of-pocket impact using claims data")
    st.markdown("- Plugin architecture for community-contributed modules")

# -----------------------------------------------------------------------------
# 6. RUN APP
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
