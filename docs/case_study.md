# Case Study: Two AI Problems Every Conversational Commerce Platform Actually Has

**Predictive merchant-health modeling and conversational-AI monitoring, built to demonstrate the two capabilities an "AI-powered operational platform for WhatsApp-first SMBs" needs beyond dashboards: knowing which merchants are about to leave, and knowing whether the AI agent talking to them is actually working.**

---

## The business question

Most analytics portfolios stop at descriptive dashboards. An AI-operations platform needs two things dashboards don't give it: a model that predicts merchant churn before it happens, and a way to monitor whether its own conversational AI agent is degrading in production. This repo builds both, as two independent but complementary modules.

## Module 1: Merchant Churn & Survival (`merchant_churn_survival/`)

**Data:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI/Kaggle) — 1,067,371 real transaction line items from a UK wholesale gift/homeware retailer, Dec 2009–Dec 2011. Framed here as a B2B SaaS analogue: this dataset's "Customer ID" is disproportionately a repeat wholesale account (a small business restocking inventory), not a one-off consumer — the same structural role a merchant subscriber plays on an AI-operations platform.

**Method:** RFM segmentation, a logistic regression churn classifier (RFM + tenure features), Kaplan-Meier survival analysis, a Cox Proportional Hazards model, and a composite Merchant Health Score blending the ML prediction with the RFM tier.

**Findings:**
- **50.8%** of accounts are churned under a 90-day-silence definition.
- The churn classifier reaches **71.5% accuracy / 0.780 ROC-AUC** on held-out data using only RFM + tenure.
- The Cox model quantifies *why*: order frequency is the strongest protective factor — a conversational-commerce nudge that gets one more order out of a slowing account measurably extends its lifetime, not just its optics.
- **1,883 accounts (32%)** score below the at-risk threshold today, representing **£579,035** in monetary value — a live worklist, not a retrospective report.
- A retention campaign saving even 25% of at-risk accounts nets an estimated **£116,591** after outreach cost (live Excel what-if model).

A data-quality finding surfaced, not hidden: 6,202 real transaction rows carry `price = 0` and represent stock write-offs/adjustments, not sales — one such row alone was a -1,560-unit adjustment that would have silently corrupted demand and churn features had it not been explicitly filtered and documented.

Full report: [`merchant_churn_survival/reports/churn_report.md`](../merchant_churn_survival/reports/churn_report.md)

## Module 2: Conversational AI Intent Monitor (`conversational_ai_monitor/`)

**Data:** a curated 20-intent subset of [CLINC150](https://github.com/clinc/oos-eval) (Larson et al., EMNLP 2019) — a real, peer-reviewed, crowd-sourced benchmark of human-written utterances (predating LLM-generated synthetic data entirely), selected for intents that map directly onto a WhatsApp commerce/ops agent's job: order management, payments/billing, complaints/security, delivery/logistics, and scheduling, plus a genuine out-of-scope class.

**Method:** TF-IDF + Logistic Regression intent classification, confidence calibration (reliability diagrams, Brier score), and a drift-monitoring pipeline with statistical alerting.

**Findings:**
- The classifier reaches **89.5–90% accuracy / ~0.90 macro F1** across 20 real intents — strong enough to auto-route inbound WhatsApp messages.
- Confidence calibration (Brier score ≈ 0.27) shows where the agent's stated confidence can and can't be trusted as a hand-off trigger.
- A drift-monitoring simulation — built entirely from real, unmodified utterances resampled into a documented, disclosed scenario (a rising share of the model's hardest intents from batch 6 onward) — correctly triggered a statistical accuracy-drop alert, the exact signal that should page a human before customers notice the agent getting worse.
- A reverse ETL pipeline (Firestore-shaped documents → structured PostgreSQL table) demonstrates the opposite data-engineering direction from typical SQL→NoSQL portfolios: making a conversational AI product's native document store queryable in plain SQL for BI.

Full report: [`conversational_ai_monitor/reports/intent_report.md`](../conversational_ai_monitor/reports/intent_report.md)

## What's real vs. constructed (stated plainly, both modules)

Every number in both reports comes from running real code against real, sourced data. Two things are explicitly constructed, and disclosed as such rather than presented as organic: (1) the 90-day churn cutoff is a standard, documented threshold choice, not a discovered fact; (2) the conversational AI module's "batch" time-ordering and one drift scenario are simulated because CLINC150 is a static benchmark with no timestamps — the underlying text and labels are entirely real.

## Recommendations

1. **Operationalize the Merchant Health Score** as a daily account-manager worklist, not a one-time report.
2. **Route retention spend using the Cox hazard ratios** — frequency-boosting nudges outperform generic re-engagement blasts.
3. **Wire the drift-alert rule into a real monitoring job** — the statistical method here (rolling accuracy vs. baseline) needs no ML sophistication to deploy, just a scheduled query.
4. **Treat confidence calibration as a hand-off policy input**, not a vanity metric — a mis-calibrated "high confidence" wrong answer is worse for trust than a correctly-flagged low-confidence hand-off.

## Tools & skills demonstrated

SQL (PostgreSQL: joins, CTEs, window functions, subqueries) · Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, lifelines) · Survival analysis (Kaplan-Meier, Cox Proportional Hazards) · NLP intent classification, confidence calibration, drift monitoring · Excel (live formulas, what-if modeling) · NoSQL/document modeling (Firestore-shaped documents) · Reverse ETL (NoSQL → structured SQL) · Live interactive dashboards (Streamlit)
