"""
ETL: Firestore-shaped documents -> structured PostgreSQL table.

This is the opposite direction from every other ETL in this portfolio
(which goes relational -> NoSQL). Here, a conversational AI product's
natural operational store (documents keyed by conversation) gets flattened
into a queryable analytics table -- exactly the pipeline a real BI team
would run nightly to make Firestore-native app data joinable with the
rest of the warehouse in plain SQL.
"""
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / "data" / "staged" / "conversations.jsonl"
PSQL = r"C:\Program Files\PostgreSQL\16\bin\psql.exe"
DB = {"host": "localhost", "port": "5432", "user": "postgres", "dbname": "smb_ai_operations"}

DDL = """
DROP TABLE IF EXISTS conversations;
CREATE TABLE conversations (
    conversation_id   VARCHAR(20) PRIMARY KEY,
    utterance         TEXT,
    true_intent       VARCHAR(40),
    category          VARCHAR(40),
    predicted_intent  VARCHAR(40),
    confidence        NUMERIC(5,4),
    is_correct        BOOLEAN,
    batch             INTEGER
);
CREATE INDEX idx_conversations_batch ON conversations(batch);
CREATE INDEX idx_conversations_category ON conversations(category);
"""


def run_sql(sql: str) -> None:
    cmd = [PSQL, "-h", DB["host"], "-p", DB["port"], "-U", DB["user"], "-d", DB["dbname"], "-c", sql]
    env = os.environ.copy()
    env.setdefault("PGPASSWORD", "postgres")
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    docs = [json.loads(line) for line in STAGED.open(encoding="utf-8")]
    df = pd.DataFrame(docs)

    tmp = ROOT / "data" / "staged" / "_conversations_flat.csv"
    df.to_csv(tmp, index=False)

    subprocess.run(
        [PSQL, "-h", DB["host"], "-p", DB["port"], "-U", DB["user"], "-d", DB["dbname"], "-c", DDL],
        env={**os.environ, "PGPASSWORD": "postgres"}, check=True,
    )
    copy_cmd = f"\\copy conversations FROM '{tmp.as_posix()}' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
    subprocess.run(
        [PSQL, "-h", DB["host"], "-p", DB["port"], "-U", DB["user"], "-d", DB["dbname"], "-c", copy_cmd],
        env={**os.environ, "PGPASSWORD": "postgres"}, check=True,
    )
    tmp.unlink()
    print(f"Loaded {len(df):,} documents into structured table 'conversations' in {DB['dbname']}")


if __name__ == "__main__":
    main()
