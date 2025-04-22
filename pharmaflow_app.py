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
    df = pd.json_normalize(data)
    return df

# 2. Fetch Medicaid NADAC pricing data
@st.cache_data(ttl=3600)
def fetch_nadac_data():
    url = "https://data.medicaid.gov/api/1/datastore/query/4d7af295-2132-55a8-b40c-d6630061f3e8/0/download?format=csv"
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

# Main app
st.title("PharmaFlow: Modeling FDA Policy Impacts on Drug Prices & Availability")

# Section 1: Drug shortages
st.markdown("## 1. Live Drug Shortage Data")
shortage_df = fetch_shortage_data(100)
if not shortage_df.empty:
    st.dataframe(shortage_df[['product_type','product_name','dosage_form','route','manufacturer','status','report_date']])
else:
    st.write("No shortage data available at the moment.")

# Section 2: NADAC pricing trends
st.markdown("## 2. Current Medicaid NADAC Pricing Trends")
nadac_df = fetch_nadac_data()
if 'NDC' in nadac_df.columns and 'NADAC Per Unit' in nadac_df.columns:
    st.dataframe(nadac_df[['NDC','NADAC Per Unit','As of Date']].head(10))
    baseline_price_mean = nadac_df['NADAC Per Unit'].mean()
    st.markdown(f"**Average NADAC price per unit:** ${baseline_price_mean:.2f}")
else:
    st.write("Unexpected NADAC data format.")

# Sidebar controls
st.sidebar.header("Policy Levers & Simulation Parameters")
new_exclusivity = st.sidebar.slider("New exclusivity period (years)", min_value=5.0, max_value=20.0, value=12.0, step=1.0)
num_competitors = st.sidebar.slider("Expected number of generic competitors", min_value=1, max_value=10, value=1, step=1)
horizon = st.sidebar.slider("Simulation horizon (years)", min_value=1, max_value=10, value=3, step=1)
n_sim = st.sidebar.number_input("Number of Monte Carlo simulations", min_value=100, max_value=10000, value=1000, step=100)

st.sidebar.markdown("### Sources")
st.sidebar.markdown("- Live drug shortage data via openFDA API citeturn0search4")
st.sidebar.markdown("- NADAC pricing data via Data.Medicaid.gov citeturn1search0")
st.sidebar.markdown("- Generic competition & price drop study (FDA) citeturn6search7")
st.sidebar.markdown("- Median generic entry delay study citeturn7search1")

# Section 3: Monte Carlo simulation
st.markdown("## 3. Monte Carlo Simulation of Price Outcomes")
entry_delay_mean = new_exclusivity
entry_delay_sd = entry_delay_mean * (3.04/14.1)  # variability based on IQR study
drop_mean = price_drop_mean(num_competitors)
drop_sd = drop_mean * 0.1

time_points = np.linspace(0, horizon, num=50)
price_ratios = np.zeros((n_sim, len(time_points)))

for i in range(n_sim):
    entry = np.random.normal(entry_delay_mean, entry_delay_sd)
    price_drop = np.random.normal(drop_mean, drop_sd)
    price_drop = np.clip(price_drop, 0, 1)
    for j, t in enumerate(time_points):
        price_ratios[i, j] = 1.0 if t < entry else 1 - price_drop

mean_price_ratio = price_ratios.mean(axis=0)
st.line_chart(pd.DataFrame({"Average Price Ratio": mean_price_ratio}, index=time_points))

# Summary at horizon
final_ratios = price_ratios[:, -1]
mean_drop = 1 - final_ratios.mean()
ci_lower, ci_upper = np.percentile(final_ratios, [2.5, 97.5])
ci_drop_lower = 1 - ci_upper
ci_drop_upper = 1 - ci_lower

st.markdown(f"### Projected Price Drop at {horizon} years")
st.markdown(f"- **Mean price drop:** {mean_drop*100:.1f}%")
st.markdown(f"- **95% CI:** [{ci_drop_lower*100:.1f}%, {ci_drop_upper*100:.1f}%]")

# Interpretation
st.markdown("### Interpretation")
st.markdown("Prices remain constant until generic entry, after which they decline according to observed industry dynamics.")

# Section 4: Caveats & next steps
st.markdown("## 4. Model Caveats & Next Steps")
st.markdown('''
- Price and entry time distributions are based on historical studies; actual outcomes may vary.
- Collaborate with ICER and Kessel Run to refine model assumptions and parameter distributions.
- Expand to distinguish small molecules vs biologics and include patent evergreening rules.
- Incorporate supply-side dynamics to proactively model drug shortages.
''')
