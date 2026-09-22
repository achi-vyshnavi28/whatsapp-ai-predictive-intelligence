# merchant_churn_survival

Predicts merchant/account churn and quantifies retention ROI using real B2B wholesale transaction data.

**Data:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI/Kaggle) — 1,067,371 real transactions, Dec 2009–Dec 2011, UK wholesale gift/homeware retailer. Most repeat customers are small businesses restocking inventory, used here as a structural analogue for merchant subscribers.

**Method:** RFM segmentation → logistic regression churn classifier → Kaplan-Meier survival curves → Cox Proportional Hazards → composite Merchant Health Score.

**Confirmed results (real run):**
```
Churn rate: 50.8%  Accuracy: 0.715  AUC: 0.780  At-risk accounts: 1,883
```

Full report: [`reports/churn_report.md`](reports/churn_report.md)

## Running it
```bash
pip install -r ../requirements.txt
psql -U postgres -c "CREATE DATABASE smb_ai_operations"
psql -U postgres -d smb_ai_operations -f sql/01_schema.sql
python python/load_data.py
python python/eda_analysis.py
python python/build_excel_model.py
```
