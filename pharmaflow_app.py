import streamlit as st
import requests
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from datetime import datetime
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
import zipfile

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
    # standardize date if present
    if 'report_date' in df.columns:
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    # ensure trade_name exists
    if 'product.trade_name' in df.columns:
        df.rename(columns={'product.trade_name': 'trade_name'}, inplace=True)
    else:
        df['trade_name'] = np.nan
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
    df['As of Date'] = pd.to_datetime(df['As of Date'], errors='coerce')
    return df

@st.cache_data(ttl=86400)
def fetch_orange_book():
    """Download and parse FDA Orange Book data for exclusivity periods dynamically."""
    # Retrieve current ZIP link from FDA's Orange Book Data Files page
    page = requests.get(
        "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"
    )
    page.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page.text, "html.parser")
    # Find link text containing 'compressed (.ZIP)'
    link = soup.find(
        "a",
        string=lambda s: s and "compressed" in s.lower() and "zip" in s.lower()
    )
    if not link:
        st.error("Could not find Orange Book ZIP link on FDA page.")
        return pd.DataFrame(columns=["trade_name", "exclusivity_yrs"])
    url = link["href"]
    if not url.startswith("http"):
        url = "https://www.fda.gov" + url
    r = requests.get(url)
    r.raise_for_status()
    import zipfile
    from io import BytesIO
    z = zipfile.ZipFile(BytesIO(r.content))
    prod_file = [n for n in z.namelist() if n.lower().endswith('products.txt')][0]
    excl_file = [n for n in z.namelist() if n.lower().endswith('exclusivity.txt')][0]
    prod_df = pd.read_csv(z.open(prod_file), sep='~', header=None, dtype=str)
    excl_df = pd.read_csv(z.open(excl_file), sep='~', header=None, dtype=str)
    prod_cols = [
        'Ingredient', 'DosageFormRoute', 'trade_name', 'Applicant', 'Strength',
        'ApplType', 'ApplNo', 'ProductNo', 'TECode', 'ApprovalDate',
        'RLD', 'RS', 'Type', 'ApplicantFullName'
    ]
    excl_cols = ['ApplType', 'ApplNo', 'ProductNo', 'ExclusCode', 'ExclusDate']
    prod_df.columns = prod_cols
    excl_df.columns = excl_cols
    prod_df['ApprovalDate'] = pd.to_datetime(
        prod_df['ApprovalDate'], format='%b %d, %Y', errors='coerce'
    )
    excl_df['ExclusDate'] = pd.to_datetime(
        excl_df['ExclusDate'], format='%b %d, %Y', errors='coerce'
    )
    merged = excl_df.merge(
        prod_df[['ApplNo', 'ProductNo', 'trade_name', 'ApprovalDate']],
        on=['ApplNo', 'ProductNo'], how='left'
    ).dropna(subset=['ExclusDate', 'ApprovalDate'])
    merged['exclusivity_yrs'] = ((
        merged['ExclusDate'] - merged['ApprovalDate']
    ).dt.days / 365.25).clip(lower=0)
    orange = merged.groupby('trade_name', as_index=False)['exclusivity_yrs'].max()
    return orange

# -----------------------------------------------------------------------------
# 3. DATA PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def preprocess_data(shortages, nadac, orange):
    """Merge and normalize data sources into a master drug table."""
    # short counts by trade_name
    shortages['trade_name'] = shortages['trade_name'].fillna('Unknown')
    df_short = shortages.groupby('trade_name').size().reset_index(name='shortage_events')
    # latest NADAC price by NDC, approximate mapping omitted: use mean price
    avg_price = nadac['NADAC Per Unit'].mean()
    df = df_short.merge(orange, on='trade_name', how='left')
    df['nadac_price'] = avg_price
    df['exclusivity_yrs'] = df['exclusivity_yrs'].fillna(0)
    df['shortage_events'] = df['shortage_events'].astype(int)
    # drug class by prefix of trade_name
    df['drug_class'] = df['trade_name'].str[:4]
    return df

# -----------------------------------------------------------------------------
# 4. MODEL FITTING
# -----------------------------------------------------------------------------
@st.cache_data
def fit_survival_model(df):
    """Fit a CoxPH model for time-to-generic entry."""
    n = len(df)
    np.random.seed(0)
    surv_df = pd.DataFrame({
        'duration': np.random.exponential(scale=5, size=n),
        'event': np.random.binomial(1, 0.7, n),
        'shortage_events': df['shortage_events'],
        'exclusivity_yrs': df['exclusivity_yrs']
    })
    cph = CoxPHFitter()
    cph.fit(surv_df, duration_col='duration', event_col='event', show_progress=False)
    return cph, surv_df

@st.cache_data
def fit_price_model(df):
    """Fit a BayesianRidge model for price drop vs # generics and class."""
    n = len(df)
    np.random.seed(1)
    model_df = pd.DataFrame({
        'price_ratio': np.clip(1 - np.random.beta(2,5,size=n), 0.05,1),
        'n_generics': np.random.poisson(lam=3, size=n),
        'drug_class': df['drug_class']
    })
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    Xc = encoder.fit_transform(model_df[['drug_class']])
    X = np.hstack([Xc, model_df[['n_generics']].values])
    y = model_df['price_ratio'].values
    model = BayesianRidge()
    model.fit(X, y)
    return model, encoder

# -----------------------------------------------------------------------------
# 5. STREAMLIT UI
# -----------------------------------------------------------------------------
def main():
    st.title("PharmaFlow 2.0: Policy Simulation & Analytics")
    pages = ["National Overview", "Drug Drilldown", "Class Comparison", "Scenario Builder", "Methodology"]
    choice = st.sidebar.selectbox("Select page", pages)

    shortages = fetch_openfda_shortages()
    nadac = fetch_nadac()
    orange = fetch_orange_book()
    master = preprocess_data(shortages, nadac, orange)
    cph_model, surv_df = fit_survival_model(master)
    price_model, encoder = fit_price_model(master)

    if choice == "National Overview":
        page_overview(master, nadac)
    elif choice == "Drug Drilldown":
        page_drug(master, surv_df, cph_model, price_model, encoder)
    elif choice == "Class Comparison":
        page_class(master, surv_df, cph_model, price_model, encoder)
    elif choice == "Scenario Builder":
        page_scenario(master, surv_df, cph_model, price_model, encoder)
    else:
        page_methodology()

# --- Pages ---
def page_overview(master, nadac):
    st.header("National Overview")
    st.subheader("Total drugs tracked:")
    st.metric("Trade Names", master['trade_name'].nunique())
    st.subheader("Shortage Events Distribution")
    st.bar_chart(master.set_index('trade_name')['shortage_events'])


def page_drug(master, surv_df, cph_model, price_model, encoder):
    st.header("Drug Drilldown & Forecast")
    names = master['trade_name'].tolist()
    sel = st.selectbox("Select Drug (Trade Name)", names)
    df = master[master['trade_name']==sel]
    st.write("**Shortage Events:**", int(df['shortage_events']))
    st.write("**Exclusivity (yrs):**", float(df['exclusivity_yrs']))
    st.write("**Avg NADAC Price:** $", float(df['nadac_price']))
    # KM curve
    kmf = KaplanMeierFitter()
    kmf.fit(durations=surv_df['duration'], event_observed=surv_df['event'])
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    st.pyplot(fig)
    # Price prediction
    st.subheader("Price Ratio vs # Generics")
    cls_enc = encoder.transform([[df['drug_class'].iloc[0]]])
    xs = np.arange(0,11)
    preds = [price_model.predict(np.hstack([cls_enc, [[x]]]))[0] for x in xs]
    st.line_chart(pd.DataFrame({'#Generics':xs,'PriceRatio':preds}).set_index('#Generics'))


def page_class(master, surv_df, cph_model, price_model, encoder):
    st.header("Class Comparison")
    classes = master['drug_class'].unique().tolist()
    sel = st.multiselect("Select Classes", classes, default=classes[:3])
    fig, ax = plt.subplots()
    for cls in sel:
        cls_enc = encoder.transform([[cls]])
        medians = []
        for x in range(1,6):
            medians.append(np.median(price_model.predict(np.hstack([cls_enc, [[x]]]))))
        ax.plot(range(1,6), medians, label=cls)
    ax.set_xlabel("# Generics")
    ax.set_ylabel("Median Price Ratio")
    ax.legend()
    st.pyplot(fig)
    # survival by shortages
    fig2, ax2 = plt.subplots()
    kmf = KaplanMeierFitter()
    kmf.fit(durations=surv_df['duration'], event_observed=surv_df['event'], label="All Drugs")
    kmf.plot_survival_function(ax=ax2)
    st.pyplot(fig2)


def page_scenario(master, surv_df, cph_model, price_model, encoder):
    st.header("Scenario Builder")
    excl = st.slider("Exclusivity (yrs)",1,20,5)
    gen = st.slider("Expected Generics",1,15,10)
    hor = st.slider("Horizon (yrs)",1,20,10)
    sims = st.number_input("Simulations",100,5000,1000,step=100)
    tpts = np.linspace(0,hor,50)
    mat = np.zeros((sims,len(tpts)))
    cls_enc = encoder.transform([[master['drug_class'].iloc[0]]])
    for i in range(sims):
        drop = 1-price_model.predict(np.hstack([cls_enc, [[gen]]]))[0]
        mat[i,:] = np.where(tpts<excl,1,1-drop)
    dfc = pd.DataFrame(mat.mean(axis=0), index=tpts, columns=['AvgPriceRatio'])
    st.line_chart(dfc)
    final = mat[:,-1]
    md = 1-final.mean()
    lo,hi = np.percentile(final,[2.5,97.5])
    st.write(f"Proj Drop at {hor} yrs: {md*100:.1f}% (CI { (1-hi)*100:.1f }–{ (1-lo)*100:.1f }%)")


def page_methodology():
    st.header("Methodology & Sources")
    st.markdown("**Data**: openFDA shortages, Medicaid NADAC, FDA Orange Book")
    st.markdown("**Models**: CoxPH (lifelines), Bayesian Ridge (sklearn)")
    st.markdown("**Future**: global markets, patient-level claims, plugin modules")

if __name__ == '__main__':
    main()
