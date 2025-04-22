import streamlit as st
import requests
import pandas as pd
import numpy as np
from io import StringIO

# 1. Fetch live drug shortage data from openFDA
@st.cache_data(ttl=3600)
def fetch_shortage_data(limit=100):
    url = f"https://api.fda.gov/drug/shortages.json?limit={limit}"
    r = requests.get(url)
    data = r.json().get('results', [])
    df = pd.json_normalize(data, sep='.')
    return df

# 2. Fetch Medicaid NADAC pricing data
@st.cache_data(ttl=3600)
def fetch_nadac_data():
    url = (
        "https://data.medicaid.gov/api/1/datastore/query/"
        "4d7af295-2132-55a8-b40c-d6630061f3e8/0/download?format=csv"
    )
    r = requests.get(url)
    csv = StringIO(r.content.decode('utf-8'))
    df = pd.read_csv(csv)
    return df

# Helper for price drop by number of competitors
def price_drop_mean(n):
    if n == 1:
        return 0.39
    elif n == 2:
        return 0.54
    elif n == 3:
        return 0.65
    elif n == 4:
        return 0.79
    else:
        return 0.95

st.title("PharmaFlow: Modeling FDA Policy Impacts on Drug Prices & Availability")

# === Section 1: Live Drug Shortage Data ===
st.markdown("## 1. Live Drug Shortage Data")
shortage_df = fetch_shortage_data(100)
if not shortage_df.empty:
    cols = shortage_df.columns

    # auto‑detect relevant fields
    name_col   = next((c for c in cols if "product" in c.lower() and "name" in c.lower()), None)
    dosage_col = next((c for c in cols if "dosage" in c.lower()), None)
    route_col  = next((c for c in cols if "route" in c.lower()), None)
    manu_col   = next((c for c in cols if "manufacturer" in c.lower()), None)
    date_col   = next(
        (c for c in cols if "report_date" in c.lower() or c.lower().endswith("date")), None
    )

    display_cols = [c for c in (name_col, dosage_col, route_col, manu_col, date_col) if c]
    st.write("**Displayed columns:**", display_cols)
    st.dataframe(shortage_df[display_cols])
else:
    st.write("No shortage data available at the moment.")

# === Section 2: NADAC pricing trends ===
st.markdown("## 2. Current Medicaid NADAC Pricing Trends")
nadac_df = fetch_nadac_data()
if {"NDC", "NADAC Per Unit", "As of Date"}.issubset(nadac_df.columns):
    st.dataframe(nadac_df[["NDC", "NADAC Per Unit", "As of Date"]].head(10))
    baseline = nadac_df["NADAC Per Unit"].mean()
    st.markdown(f"**Average NADAC price per unit:** ${baseline:.2f}")
else:
    st.write("Unexpected NADAC data format.")

# === Sidebar: Policy levers & data sources ===
st.sidebar.header("Policy Levers & Simulation Parameters")
new_exclusivity = st.sidebar.slider("New exclusivity (years)", 5.0, 20.0, 12.0, 1.0)
num_competitors = st.sidebar.slider("Number of generic competitors", 1, 10, 1, 1)
horizon         = st.sidebar.slider("Simulation horizon (years)", 1, 10, 3, 1)
n_sim           = st.sidebar.number_input("Monte Carlo simulations", 100, 10000, 1000, 100)

st.sidebar.markdown("### Data Sources")
st.sidebar.markdown(
    "- FDA Drug Shortages API: https://open.fda.gov/apis/drug/shortage/"
)
st.sidebar.markdown(
    "- Medicaid NADAC pricing CSV: https://data.medicaid.gov/resource/4d7af295.csv"
)
st.sidebar.markdown(
    "- Generic competition & price-drop meta-analysis (FDA study)"
)
st.sidebar.markdown(
    "- Generic-entry delay industry report"
)

# === Section 3: Monte Carlo Simulation ===
st.markdown("## 3. Monte Carlo Simulation of Price Outcomes")
entry_mean = new_exclusivity
entry_sd   = entry_mean * (3.04 / 14.1)
drop_mean  = price_drop_mean(num_competitors)
drop_sd    = drop_mean * 0.1

time_pts = np.linspace(0, horizon, 50)
price_rat = np.zeros((n_sim, len(time_pts)))

for i in range(n_sim):
    entry     = np.random.normal(entry_mean, entry_sd)
    drop_rate = np.clip(np.random.normal(drop_mean, drop_sd), 0, 1)
    price_rat[i] = [1.0 if t < entry else 1 - drop_rate for t in time_pts]

mean_ratio = price_rat.mean(axis=0)
st.line_chart(pd.DataFrame({"Avg Price Ratio": mean_ratio}, index=time_pts))

# Confidence intervals at horizon
final = price_rat[:, -1]
mean_drop = 1 - final.mean()
lower, upper = np.percentile(final, [2.5, 97.5])
ci_lo = 1 - upper
ci_hi = 1 - lower

st.markdown(f"### Projected Price Drop at {horizon} years")
st.markdown(f"- **Mean drop:** {mean_drop*100:.1f}%")
st.markdown(f"- **95% CI:** [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

# === Section 4: Caveats & Next Steps ===
st.markdown("## 4. Model Caveats & Next Steps")
st.markdown("""
- Distributions are based on historical studies; real outcomes may vary.
- Partner with ICER & Kessel Run to refine parameters.
- Distinguish small molecules vs biologics; model patent evergreening.
- Add supply‑side dynamics to forecast shortages proactively.
""")
