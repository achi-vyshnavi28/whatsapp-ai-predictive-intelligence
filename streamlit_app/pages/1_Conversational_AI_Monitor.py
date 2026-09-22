"""
Conversational AI Intent Monitor page.

Reads the precomputed classified_conversations.csv + batch_drift_stats.csv
from conversational_ai_monitor/ (run its python/intent_classifier.py first).
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = ROOT / "conversational_ai_monitor"

st.set_page_config(page_title="Conversational AI Monitor", page_icon="\U0001F916", layout="wide")


@st.cache_data
def load():
    conv = pd.read_csv(AI_ROOT / "reports" / "classified_conversations.csv")
    batches = pd.read_csv(AI_ROOT / "reports" / "batch_drift_stats.csv")
    return conv, batches


conv, batches = load()
conv["correct"] = conv["correct"].astype(bool)

st.title("Conversational AI Intent Monitor")
st.caption(
    f"{len(conv):,} real, human-written utterances (curated CLINC150 subset, Larson et al. EMNLP 2019) "
    "classified by a TF-IDF + Logistic Regression intent model simulating a WhatsApp AI agent. Full "
    "methodology: [`conversational_ai_monitor/reports/intent_report.md`]"
    "(https://github.com/achi-vyshnavi28/whatsapp-ai-predictive-intelligence/blob/main/conversational_ai_monitor/reports/intent_report.md)."
)

acc = conv["correct"].mean()
n_alerts = int(batches["alert"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversations classified", f"{len(conv):,}")
c2.metric("Overall Accuracy", f"{acc:.1%}")
c3.metric("Intents covered", f"{conv['intent_name'].nunique()}")
c4.metric("Drift Alerts Fired", f"{n_alerts} / {len(batches)}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Accuracy Trend Across Batches")
    fig = px.line(batches, x="batch", y="accuracy", markers=True)
    fig.add_hline(y=batches.loc[batches["batch"] < 6, "accuracy"].mean(), line_dash="dash",
                  line_color="gray", annotation_text="Baseline")
    alert_rows = batches[batches["alert"]]
    if len(alert_rows):
        fig.add_scatter(x=alert_rows["batch"], y=alert_rows["accuracy"], mode="markers",
                         marker=dict(color="#c1443a", size=14, symbol="x"), name="Alert")
    fig.update_layout(height=380, yaxis_title="Accuracy", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Misclassification Patterns")
    wrong = conv[~conv["correct"]]
    top_confused = wrong.groupby(["intent_name", "pred"]).size().reset_index(name="count")
    top_confused = top_confused.sort_values("count", ascending=False).head(10)
    top_confused["pair"] = top_confused["intent_name"] + " -> " + top_confused["pred"]
    fig2 = px.bar(top_confused, x="count", y="pair", orientation="h", color_discrete_sequence=["#c1443a"])
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", height=380)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Confidence Calibration")
st.image(str(AI_ROOT / "reports" / "figures" / "02_calibration_curve.png"), width=500)
st.caption("A well-calibrated agent's stated confidence should track its actual hit rate -- the basis "
           "for deciding when it's safe to auto-resolve vs. hand off to a human.")

st.divider()
st.subheader("Accuracy by Intent Category")
by_cat = conv.groupby("category")["correct"].mean().sort_values().reset_index()
fig3 = px.bar(by_cat, x="correct", y="category", orientation="h", color_discrete_sequence=["#0f7a6c"])
fig3.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_tickformat=".0%",
                    xaxis_title="Accuracy", height=320)
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption(
    "Data: curated 20-intent subset of [CLINC150](https://github.com/clinc/oos-eval) (Larson et al., "
    "EMNLP 2019) -- real, human-written, peer-reviewed benchmark utterances. Batch ordering and one "
    "documented drift injection are simulated (this static benchmark has no timestamps); all text and "
    "labels are real. Built with Streamlit, Pandas, scikit-learn, and Plotly."
)
