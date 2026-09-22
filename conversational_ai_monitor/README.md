# conversational_ai_monitor

Simulates and monitors a WhatsApp AI agent's intent classification: real intent classifier, confidence calibration, drift detection + alerting, and a Firestore-shaped NoSQL → structured-SQL ETL pipeline.

**Data:** a curated 20-intent subset of [CLINC150](https://github.com/clinc/oos-eval) (Larson et al., EMNLP 2019) — a real, peer-reviewed, crowd-sourced benchmark of human-written utterances (not LLM-generated), selected for intents relevant to a commerce/support agent: order management, payments/billing, complaints/security, delivery/logistics, scheduling, plus out-of-scope.

**What's real vs. simulated:** every utterance's text and intent label is real. This static benchmark carries no timestamps, so the sequential "batch" ordering used for drift monitoring — plus one deliberate, disclosed shift toward the model's hardest intents from batch 6 onward, built only from real examples — is simulated to demonstrate the monitoring methodology against a realistic scenario.

**Confirmed results (real run):**
```
Accuracy: 0.895-0.900  Macro F1: ~0.90  Brier: 0.268  Alerts: 1/10
```

Full report: [`reports/intent_report.md`](reports/intent_report.md)

## Pipeline
1. `python/intent_classifier.py` — classifier, calibration, drift simulation (no credentials needed)
2. `python/build_firestore_documents.py` — stage Firestore-shaped documents locally
3. `python/etl_to_structured_table.py` — load those documents into a structured PostgreSQL table (reverse ETL direction)
4. `python/load_to_firestore.py` — optional, loads into a real Firestore project (needs your own `GOOGLE_APPLICATION_CREDENTIALS`)

## Running it
```bash
pip install -r ../requirements.txt
python python/intent_classifier.py
python python/build_firestore_documents.py
python python/etl_to_structured_table.py
```
