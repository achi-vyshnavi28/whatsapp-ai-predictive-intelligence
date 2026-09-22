"""
Merchant Churn & Survival dashboard (home page).

Reads the precomputed account_scores.csv from merchant_churn_survival/
(run its eda_analysis.py first) -- no database required to run this app.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CHURN_ROOT = ROOT / "merchant_churn_survival"

st.set_page_config(page_title="Merchant Churn & Survival", page_icon="\U0001F4C9", layout="wide")


@st.cache_data
def load():
    return pd.read_csv(CHURN_ROOT / "reports" / "account_scores.csv")


df = load()

st.title("Merchant Churn & Survival Intelligence")
st.caption(
    f"{len(df):,} real B2B accounts (Online Retail II dataset, UK wholesale gift/homeware retailer). "
    "RFM segmentation, a logistic-regression churn classifier, Kaplan-Meier survival analysis, and a "
    "composite Merchant Health Score. Full methodology: "
    "[`merchant_churn_survival/reports/churn_report.md`](https://github.com/achi-vyshnavi28/whatsapp-ai-predictive-intelligence/blob/main/merchant_churn_survival/reports/churn_report.md)."
)

churn_rate = df["churned"].mean()
at_risk = df[df["health_score"] < 30]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accounts", f"{len(df):,}")
c2.metric("Churn Rate", f"{churn_rate:.1%}")
c3.metric("At-Risk Accounts (score < 30)", f"{len(at_risk):,}")
c4.metric("Value at Risk (GBP)", f"£{at_risk['monetary'].sum():,.0f}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Health Score Distribution")
    fig = px.histogram(df, x="health_score", nbins=30, color_discrete_sequence=["#0f7a6c"])
    fig.add_vline(x=30, line_dash="dash", line_color="#c1443a", annotation_text="At-risk threshold")
    fig.update_layout(height=380, xaxis_title="Health Score (0-100)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Accounts by RFM Segment")
    seg_counts = df["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "accounts"]
    fig2 = px.bar(seg_counts, x="accounts", y="segment", orientation="h", color_discrete_sequence=["#0f7a6c"])
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", height=380)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Survival Curve: Merchant Account Retention")
st.image(str(CHURN_ROOT / "reports" / "figures" / "03_survival_curve.png"), width=700)

st.divider()
st.subheader("At-Risk Merchant Worklist (lowest health scores)")
st.dataframe(
    at_risk.sort_values("health_score")[["customer_id", "health_score", "churn_probability", "segment",
                                          "recency_days", "frequency", "monetary"]].head(25),
    hide_index=True, use_container_width=True,
)

st.divider()
st.caption(
    "Data source: [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) "
    "(UCI ML Repository / Kaggle). Built with Streamlit, Pandas, scikit-learn, and lifelines."
)
