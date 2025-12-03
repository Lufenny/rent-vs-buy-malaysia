import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Buy vs Rent — Full Workflow", layout="wide")

st.title("🏠 Buy vs Rent Malaysia — Full Workflow Dashboard")

# ------------------------------------------------------------
# 1. EXPECTED OUTCOMES
# ------------------------------------------------------------
st.header("🎯 Expected Outcomes")

st.markdown("""
- Clear buy vs rent insights for Malaysia  
- Multi-city comparison (KL, Penang, Johor)  
- Behavioral-adjusted investment modelling  
- Long-term Monte Carlo projections  
- Interactive Streamlit dashboard  
- Final cleaned dataset: **data.csv**
""")

st.markdown("---")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
DATA_DIR = "../data"

files = {
    'Kuala Lumpur': f"{DATA_DIR}/kuala_lumpur_housing.csv",
    'Penang': f"{DATA_DIR}/penang_housing.csv",
    'Johor Bahru': f"{DATA_DIR}/johor_housing.csv",
}

@st.cache_data
def load_all():
    dfs = []
    for city, path in files.items():
        try:
            d = pd.read_csv(path, parse_dates=['date'])
            d['city'] = city
            dfs.append(d)
        except Exception as e:
            st.warning(f"Could not load {city}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_all = load_all()


# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.header("Controls")
cities = st.sidebar.multiselect(
    "Select cities",
    options=df_all['city'].unique() if not df_all.empty else [],
    default=['Kuala Lumpur']
)

if not df_all.empty:
    date_range = st.sidebar.date_input(
        "Date range",
        [df_all['date'].min(), df_all['date'].max()]
    )

# ------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------
df = df_all.copy()
if not df.empty and cities:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df['city'].isin(cities)) & (df['date'] >= start) & (df['date'] <= end)]
else:
    st.error("No data available.")


# ------------------------------------------------------------
# 2. EDA SECTION
# ------------------------------------------------------------
st.header("📊 1. Exploratory Data Analysis (EDA)")

sns.set_style("whitegrid")

st.title("📊 Exploratory Data Analysis (EDA)")
st.write("Upload your datasets below to explore Malaysia’s property market, OPR trends, and EPF returns.")


# -------------------------------------------------------------------
# USER UPLOAD SECTION
# -------------------------------------------------------------------
st.sidebar.header("📂 Upload Your Datasets")

property_file = st.sidebar.file_uploader("Upload Property Growth CSV", type=["csv"])
epf_file = st.sidebar.file_uploader("Upload EPF Returns CSV", type=["csv"])
opr_file = st.sidebar.file_uploader("Upload OPR History CSV", type=["csv"])


def safe_load(file, parse_dates=None):
    try:
        return pd.read_csv(file, parse_dates=parse_dates)
    except:
        st.error("Error reading file. Please ensure it's a valid CSV.")
        return None


# Load datasets
df_property = safe_load(property_file)
df_epf = safe_load(epf_file)
df_opr = safe_load(opr_file, parse_dates=["date"])


# -------------------------------------------------------------------
# IF NO DATA LOADED
# -------------------------------------------------------------------
if not property_file and not epf_file and not opr_file:
    st.info("""
    ### Please upload your datasets to begin EDA.

    Expected format:
    #### 🏠 Property Price Growth
    - `year`  
    - `growth` (% annual growth)

    #### 🏦 EPF Returns
    - `year`  
    - `return` (% EPF annual dividend)

    #### 💰 OPR Rates
    - `date`  
    - `opr` (% OPR level)

    Once uploaded, interactive charts will appear here.
    """)
    st.stop()


# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Property Growth",
    "🏦 EPF Returns",
    "💰 OPR Analysis",
    "🔗 Relationships"
])


# -------------------------------------------------------------------
# TAB 1: PROPERTY GROWTH
# -------------------------------------------------------------------
with tab1:
    if df_property is None:
        st.warning("Please upload your Property Growth CSV.")
    else:
        st.header("🏠 Property Price Growth")

        # Year Filter
        years = st.slider(
            "Select Year Range:",
            int(df_property.year.min()),
            int(df_property.year.max()),
            (int(df_property.year.min()), int(df_property.year.max()))
        )

        dfp = df_property[
            (df_property["year"] >= years[0]) &
            (df_property["year"] <= years[1])
        ]

        # Histogram
        st.subheader("Distribution of Annual Growth")
        fig, ax = plt.subplots()
        sns.histplot(dfp["growth"], kde=True, ax=ax)
        ax.set_xlabel("Annual Property Price Growth (%)")
        st.pyplot(fig)

        # Boxplot
        st.subheader("Volatility (Boxplot)")
        fig, ax = plt.subplots()
        sns.boxplot(x=dfp["growth"], ax=ax)
        st.pyplot(fig)

        # Rolling Volatility
        st.subheader("10-year Rolling Volatility")
        dfp_rolling = dfp.set_index("year")["growth"].rolling(10).std()

        fig, ax = plt.subplots()
        dfp_rolling.plot(ax=ax)
        ax.set_ylabel("Rolling Std Dev")
        st.pyplot(fig)


# -------------------------------------------------------------------
# TAB 2: EPF RETURNS
# -------------------------------------------------------------------
with tab2:
    if df_epf is None:
        st.warning("Please upload your EPF Returns CSV.")
    else:
        st.header("🏦 EPF Dividend Rates")

        # Line Chart
        st.subheader("EPF Dividend Rates Over Time")
        fig, ax = plt.subplots()
        ax.plot(df_epf["year"], df_epf["return"])
        ax.set_xlabel("Year")
        ax.set_ylabel("EPF Return (%)")
        st.pyplot(fig)

        # Histogram
        st.subheader("Distribution of EPF Returns")
        fig, ax = plt.subplots()
        sns.histplot(df_epf["return"], kde=True, ax=ax)
        st.pyplot(fig)

        # Boxplot
        st.subheader("Volatility (Boxplot)")
        fig, ax = plt.subplots()
        sns.boxplot(x=df_epf["return"], ax=ax)
        st.pyplot(fig)


# -------------------------------------------------------------------
# TAB 3: OPR ANALYSIS
# -------------------------------------------------------------------
with tab3:
    if df_opr is None:
        st.warning("Please upload your OPR History CSV.")
    else:
        st.header("💰 Overnight Policy Rate (OPR)")

        df_opr["year"] = df_opr["date"].dt.year
        df_opr["opr_change"] = df_opr["opr"].diff()

        highlight = st.checkbox("Highlight COVID Crash (2020–2021)")

        # Line chart
        st.subheader("OPR Trend")
        fig, ax = plt.subplots()
        ax.plot(df_opr["date"], df_opr["opr"])

        if highlight:
            covid = df_opr[df_opr["year"].isin([2020, 2021])]
            ax.scatter(covid["date"], covid["opr"], color="red", label="COVID OPR Crash")
            ax.legend()

        st.pyplot(fig)

        # Histogram of change
        st.subheader("Distribution of OPR Monthly Change")
        fig, ax = plt.subplots()
        sns.histplot(df_opr["opr_change"].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

        # Boxplot
        st.subheader("Volatility (Boxplot)")
        fig, ax = plt.subplots()
        sns.boxplot(x=df_opr["opr_change"], ax=ax)
        st.pyplot(fig)


# -------------------------------------------------------------------
# TAB 4: RELATIONSHIPS
# -------------------------------------------------------------------
with tab4:
    if df_property is None or df_opr is None:
        st.warning("Upload both Property and OPR datasets to view relationships.")
    else:
        st.header("🔗 Relationship Between OPR & Property Growth")

        # Merge datasets
        df_opr_yearly = df_opr.groupby("year")["opr"].mean().reset_index()
        merged = pd.merge(df_property, df_opr_yearly, on="year", how="inner")

        # Scatterplot
        st.subheader("Scatterplot: OPR vs Property Growth")
        fig, ax = plt.subplots()
        sns.regplot(x=merged["opr"], y=merged["growth"], ax=ax)
        ax.set_xlabel("Average OPR (%)")
        ax.set_ylabel("Property Price Growth (%)")
        st.pyplot(fig)

        # Correlation matrix
        st.subheader("Correlation Matrix")
        st.write(merged[["opr", "growth"]].corr())



# ------------------------------------------------------------
# 3. ANALYSIS
# ------------------------------------------------------------
st.header("📈 2. Analysis")

st.markdown("""
- Compare city growth  
- Evaluate EPF vs housing performance  
- Study OPR effects  
- Behavioral finance: savings & reinvestment discipline  
""")

if {'opr','property_index'}.issubset(df.columns):
    st.subheader("OPR vs Property Growth")
    df['year'] = df['date'].dt.year
    agg = df.groupby(['city','year']).agg({'property_index':'last','opr':'mean'})
    agg['growth'] = agg.groupby('city')['property_index'].pct_change()

    fig, ax = plt.subplots(figsize=(6,3))
    for city, g in agg.dropna().groupby('city'):
        ax.scatter(g['opr'], g['growth'], label=city)
    ax.legend()
    st.pyplot(fig)

st.markdown("---")


# ------------------------------------------------------------
# 4. DATA PROCESSING
# ------------------------------------------------------------
st.header("🧹 3. Data Processing")

st.markdown("""
- Merge multiple city datasets  
- Handle missing values  
- Convert dates  
- Engineering: monthly growth, real growth, reinvest probability  
""")

st.write(df.head())

st.markdown("---")


# ------------------------------------------------------------
# 5. MODELLING
# ------------------------------------------------------------
st.header("🧮 4. Modelling")

st.markdown("""
### Monte Carlo Model  
- Geometric Brownian motion  
- 10% volatility  
- Optional periodic crash  
- Reinvestment probability  
""")

st.sidebar.markdown("### Simulation Settings")
years = st.sidebar.slider("Years", 5, 40, 20)
sims = st.sidebar.slider("Simulations", 50, 1000, 200)
crash = st.sidebar.checkbox("Include crash every 5 years?", True)
reinvest_p = st.sidebar.slider("Reinvestment probability", 0.0, 1.0, 0.8)

# Simple Monte Carlo example
if not df.empty and 'property_index' in df.columns:
    last_price = df.sort_values('date')['property_index'].iloc[-1]
    returns = df['returns'].dropna()
    mu = returns.mean() * 12
    sigma = returns.std() * np.sqrt(12)

    T = years * 12
    dt = 1/12
    paths = np.zeros((sims, T))
    paths[:, 0] = last_price

    rng = np.random.default_rng(123)

    for t in range(1, T):
        z = rng.normal(size=sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

        if crash and t % (5*12) == 0:
            paths[:, t] *= 0.8

    median_path = np.median(paths, axis=0)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(median_path)
    st.pyplot(fig)

st.markdown("---")


# ------------------------------------------------------------
# 6. RESULTS INTERPRETATION
# ------------------------------------------------------------
st.header("📘 5. Results Interpretation")

st.markdown("""
- When buying is better  
- When renting + investing wins  
- Impact of OPR  
- City comparison  
- Behavioral influence  
""")

st.markdown("---")


# ------------------------------------------------------------
# 7. DEPLOYMENT
# ------------------------------------------------------------
st.header("🚀 6. Deployment")

st.markdown("""
- Streamlit interactive dashboard  
- CSV export  
- Scenario testing  
""")

st.markdown("---")


# ------------------------------------------------------------
# 8. FINAL DATASET EXPORT
# ------------------------------------------------------------
st.header("📁 7. Final Dataset (data.csv)")

if not df.empty:
    st.download_button(
        "Download data.csv",
        df.to_csv(index=False),
        file_name="data.csv",
        mime="text/csv"
    )
