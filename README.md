# whatsapp-ai-predictive-intelligence

Two predictive/AI-operations projects in one repository, built to demonstrate exactly what an **AI-powered operational platform for WhatsApp-first SMBs** needs beyond dashboards: a model that predicts merchant churn before it happens, and a system that monitors whether its own conversational AI agent is still working.

**Start here:** [`docs/case_study.md`](docs/case_study.md) — business question → findings → quantified impact → recommendations for both modules.

**Live dashboard:** run `streamlit run streamlit_app/app.py` (home = churn/survival, second page = conversational AI monitor) — see Reproducing below.

## Project 1: `merchant_churn_survival/`

Predicts which merchant accounts are about to go dormant, and quantifies the revenue at stake — using real B2B wholesale transaction data ([Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI/Kaggle, 1.07M real transactions).

| Skill | Where |
|---|---|
| Complex SQL: joins, CTEs, window functions, subqueries | [`sql/02_analysis_queries.sql`](merchant_churn_survival/sql/02_analysis_queries.sql) — 9 queries incl. cohort retention, RFM, churn labeling |
| Python EDA (Pandas, NumPy, Matplotlib/Seaborn) | [`python/eda_analysis.py`](merchant_churn_survival/python/eda_analysis.py) |
| **Churn prediction** (explicitly named in the JD) | Logistic regression, 71.5% accuracy / 0.780 ROC-AUC |
| Survival analysis (more advanced than standard churn ML) | Kaplan-Meier curves + Cox Proportional Hazards (`lifelines`) |
| Cohort/retention analysis | Monthly cohort retention curves (SQL Q2) |
| Composite Merchant Health Score | Blends churn probability + RFM into one worklist number |
| Excel modeling | [`excel/churn_retention_model.xlsx`](merchant_churn_survival/excel/churn_retention_model.xlsx) — live formulas, retention-ROI what-if |
| Anomaly detection | Documented data-quality find: 6,202 stock-adjustment rows with `price=0` that would have corrupted demand/churn features if unfiltered |
| Live dashboard | `streamlit_app/app.py` — health scores, survival curve, at-risk merchant worklist |

## Project 2: `conversational_ai_monitor/`

Simulates and monitors a WhatsApp AI agent's intent classification — using [CLINC150](https://github.com/clinc/oos-eval) (Larson et al., EMNLP 2019), a real, peer-reviewed, human-written benchmark (not LLM-generated), curated to 20 commerce/support-relevant intents + out-of-scope.

| Skill | Where |
|---|---|
| NLP intent classification (nice-to-have) | TF-IDF + Logistic Regression, 89.5% accuracy / ~0.90 macro F1 across 20 intents |
| Chatbot performance tracking (nice-to-have) | Confusion matrix, per-intent precision/recall/F1, misclassification-pattern analysis |
| Confidence calibration | Reliability diagram + Brier score — a technique neither sibling repo uses |
| Drift monitoring + alerting | Rolling accuracy vs. baseline, statistical alert rule, disclosed simulated scenario |
| Firestore / NoSQL (nice-to-have) | [`python/build_firestore_documents.py`](conversational_ai_monitor/python/build_firestore_documents.py) + [`load_to_firestore.py`](conversational_ai_monitor/python/load_to_firestore.py) |
| ETL (nice-to-have) | [`etl_to_structured_table.py`](conversational_ai_monitor/python/etl_to_structured_table.py) — NoSQL documents → structured PostgreSQL table (the reverse direction from typical SQL→NoSQL portfolios) |
| SQL on the ETL'd table | [`sql/01_analysis_queries.sql`](conversational_ai_monitor/sql/01_analysis_queries.sql) — window-function drift detection, subqueries |
| Live dashboard | `streamlit_app/pages/1_Conversational_AI_Monitor.py` — accuracy trend, drift flags, misclassification patterns |

## Data sources

- **Online Retail II**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) / Kaggle. Real UK wholesale gift/homeware retailer transactions, Dec 2009–Dec 2011. Not synthetic.
- **CLINC150**: [github.com/clinc/oos-eval](https://github.com/clinc/oos-eval) (also listed on Kaggle). Real, crowd-sourced, peer-reviewed intent-classification benchmark (Larson et al., EMNLP 2019). Not synthetic, not LLM-generated.

## Repo structure

```
merchant_churn_survival/
  sql/            01_schema.sql, 02_analysis_queries.sql
  python/         load_data.py, eda_analysis.py, build_excel_model.py
  excel/          churn_retention_model.xlsx
  data/raw/       online_retail.csv.gz
  reports/        churn_report.md + figures/ + account_scores.csv (auto-generated)
conversational_ai_monitor/
  sql/            01_analysis_queries.sql
  python/         intent_classifier.py, build_firestore_documents.py,
                   load_to_firestore.py, etl_to_structured_table.py
  data/raw/       clinc150_commerce_intents.csv
  data/staged/    conversations.jsonl (Firestore-shaped documents)
  reports/        intent_report.md + figures/ (auto-generated)
streamlit_app/    app.py (churn/survival) + pages/1_Conversational_AI_Monitor.py
docs/             case_study.md
```

## Reproducing this locally

**Module 1 — Merchant Churn & Survival:**
```bash
pip install -r requirements.txt
psql -U postgres -c "CREATE DATABASE smb_ai_operations"
psql -U postgres -d smb_ai_operations -f merchant_churn_survival/sql/01_schema.sql
python merchant_churn_survival/python/load_data.py
python merchant_churn_survival/python/eda_analysis.py
python merchant_churn_survival/python/build_excel_model.py
```

**Module 2 — Conversational AI Monitor:**
```bash
python conversational_ai_monitor/python/intent_classifier.py
python conversational_ai_monitor/python/build_firestore_documents.py
python conversational_ai_monitor/python/etl_to_structured_table.py   # needs the same local Postgres

# Optional — needs your own Firebase project (never handled by this code):
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/serviceAccountKey.json"
python conversational_ai_monitor/python/load_to_firestore.py
```

**Live dashboard (either module, no database required):**
```bash
streamlit run streamlit_app/app.py
```

## Notes on scope

This repo deliberately covers churn/survival prediction and conversational-AI monitoring — the two capabilities its sibling repos ([whatsapp-order-ops-analytics](https://github.com/achi-vyshnavi28/whatsapp-order-ops-analytics), [smb-merchant-funnel-analytics](https://github.com/achi-vyshnavi28/smb-merchant-funnel-analytics)) don't touch — rather than adding a third layer of "operational analytics" on top of order-ops and acquisition-funnel work already covered elsewhere. Power BI dashboards are demonstrated in both sibling repos; this repo's live dashboard is Streamlit, consistent with the rest of the portfolio's deployment pattern.
