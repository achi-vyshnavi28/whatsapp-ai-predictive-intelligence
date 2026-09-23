# whatsapp-ai-predictive-intelligence

Two predictive/AI projects exploring what dashboards alone don't answer: which merchants are quietly about to leave, and whether the AI agent handling customer conversations is actually still working.

**Start here:** [`docs/case_study.md`](docs/case_study.md) — business question → findings → quantified impact → recommendations for both modules.

**Live dashboard:** run `streamlit run streamlit_app/app.py` (home = churn/survival, second page = conversational AI monitor) — see Reproducing below.

## Project 1: `merchant_churn_survival/`

Predicts which merchant accounts are about to go dormant, and quantifies the revenue at stake — using real B2B wholesale transaction data ([Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI/Kaggle, 1.07M real transactions).

- **SQL** — 9 PostgreSQL queries covering cohort retention, RFM, and churn labeling using joins, CTEs, window functions, and subqueries ([`sql/`](merchant_churn_survival/sql/)).
- **Python** — EDA with Pandas, NumPy, Matplotlib, and Seaborn ([`python/eda_analysis.py`](merchant_churn_survival/python/eda_analysis.py)).
- **Churn prediction** — logistic regression, 71.5% accuracy / 0.780 ROC-AUC.
- **Survival analysis** — Kaplan-Meier curves and a Cox Proportional Hazards model (`lifelines`).
- **Cohort/retention analysis** — monthly cohort retention curves (SQL Q2).
- **Composite Merchant Health Score** — blends churn probability and RFM into one worklist number.
- **Excel modeling** — live formulas for a retention-ROI what-if model ([`excel/churn_retention_model.xlsx`](merchant_churn_survival/excel/churn_retention_model.xlsx)).
- **Anomaly detection** — a documented data-quality find: 6,202 stock-adjustment rows with `price=0` that would have corrupted demand/churn features if left unfiltered.
- **Live dashboard** — `streamlit_app/app.py`, showing health scores, the survival curve, and an at-risk merchant worklist.

## Project 2: `conversational_ai_monitor/`

Simulates and monitors a WhatsApp AI agent's intent classification — using [CLINC150](https://github.com/clinc/oos-eval) (Larson et al., EMNLP 2019), a real, peer-reviewed, human-written benchmark (not LLM-generated), curated to 20 commerce/support-relevant intents plus an out-of-scope class.

- **NLP intent classification** — TF-IDF + Logistic Regression, 89.5% accuracy / ~0.90 macro F1 across 20 intents.
- **Chatbot performance tracking** — confusion matrix, per-intent precision/recall/F1, and misclassification-pattern analysis.
- **Confidence calibration** — a reliability diagram and Brier score.
- **Drift monitoring and alerting** — rolling accuracy vs. baseline with a statistical alert rule, against a disclosed simulated drift scenario.
- **Firestore / NoSQL** — [`python/build_firestore_documents.py`](conversational_ai_monitor/python/build_firestore_documents.py) and [`load_to_firestore.py`](conversational_ai_monitor/python/load_to_firestore.py).
- **ETL** — [`etl_to_structured_table.py`](conversational_ai_monitor/python/etl_to_structured_table.py), loading NoSQL documents into a structured PostgreSQL table (the reverse direction from a typical SQL-to-NoSQL pipeline).
- **SQL** — [`sql/01_analysis_queries.sql`](conversational_ai_monitor/sql/01_analysis_queries.sql), window-function drift detection and subqueries against the ETL'd table.
- **Live dashboard** — `streamlit_app/pages/1_Conversational_AI_Monitor.py`, showing accuracy trend, drift flags, and misclassification patterns.

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
