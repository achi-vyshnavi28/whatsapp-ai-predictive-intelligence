"""
Loads the Online Retail II dataset (Kaggle/UCI, real UK wholesale-gift-
retailer transactions, Dec 2009-Dec 2011) into the smb_ai_operations
PostgreSQL database.

The raw export (data/raw/online_retail.csv.gz) is one flat table; this
script normalizes it into customers/products/transactions at load time
(same discipline as the sibling repos' schema design) and derives
`is_cancellation` from the real Invoice-prefix convention this dataset
uses ('C' prefix = a credit/cancellation of an earlier invoice).
"""
import gzip
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail.csv.gz"
STAGED = ROOT / "data" / "staged"
STAGED.mkdir(parents=True, exist_ok=True)
PSQL = r"C:\Program Files\PostgreSQL\16\bin\psql.exe"

DB = {"host": "localhost", "port": "5432", "user": "postgres", "dbname": "smb_ai_operations"}


def run_copy(table: str, csv_path: Path) -> None:
    copy_cmd = f"\\copy {table} FROM '{csv_path.as_posix()}' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
    cmd = [PSQL, "-h", DB["host"], "-p", DB["port"], "-U", DB["user"], "-d", DB["dbname"], "-c", copy_cmd]
    env = os.environ.copy()
    env.setdefault("PGPASSWORD", "postgres")
    subprocess.run(cmd, env=env, check=True)
    print(f"Loaded {table} from {csv_path.name}")


def main() -> None:
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, parse_dates=["invoicedate"])

    df = df.rename(columns={
        "invoice": "invoice", "stockcode": "stock_code", "description": "description",
        "quantity": "quantity", "invoicedate": "invoice_date", "price": "price",
        "customer_id": "customer_id", "country": "country",
    })
    df["invoice"] = df["invoice"].astype(str)
    df["is_cancellation"] = df["invoice"].str.startswith("C")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")

    # --- customers ---
    customers = (
        df.dropna(subset=["customer_id"])
        .groupby("customer_id")["country"]
        .agg(lambda s: s.mode().iat[0])
        .reset_index()
    )
    customers_path = STAGED / "customers.csv"
    customers.to_csv(customers_path, index=False)

    # --- products: most common non-null description per stock code ---
    desc = df.dropna(subset=["description"]).copy()
    desc["description"] = desc["description"].str.strip()
    products = (
        desc.groupby("stock_code")["description"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .reset_index()
    )
    # A handful of stock codes only ever appear with a null description
    missing_codes = set(df["stock_code"].unique()) - set(products["stock_code"])
    if missing_codes:
        products = pd.concat([
            products,
            pd.DataFrame({"stock_code": sorted(missing_codes), "description": None}),
        ], ignore_index=True)
    products_path = STAGED / "products.csv"
    products.to_csv(products_path, index=False)

    # --- transactions ---
    transactions = df[[
        "invoice", "stock_code", "customer_id", "country", "invoice_date",
        "quantity", "price", "is_cancellation",
    ]]
    transactions_path = STAGED / "transactions.csv"
    transactions.to_csv(transactions_path, index=False)

    run_copy("customers", customers_path)
    run_copy("products", products_path)
    run_copy("transactions", transactions_path)

    print(f"\nRows: customers={len(customers):,}  products={len(products):,}  transactions={len(transactions):,}")


if __name__ == "__main__":
    main()
