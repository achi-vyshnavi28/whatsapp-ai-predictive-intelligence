"""
Models the classified conversations as genuine Firestore-shaped documents
(collection: "conversations") -- one doc per classified utterance, staged
as JSONL. This is the NoSQL side of the module: read pattern is "give me
this conversation's classification result", not a join.

Run after intent_classifier.py (needs reports/classified_conversations.csv).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "classified_conversations.csv"
STAGED = ROOT / "data" / "staged"
STAGED.mkdir(parents=True, exist_ok=True)


def build_document(row) -> dict:
    return {
        "conversation_id": f"conv_{row.name:06d}",
        "utterance": row["text"],
        "true_intent": row["intent_name"],
        "category": row["category"],
        "predicted_intent": row["pred"],
        "confidence": round(float(row["confidence"]), 4),
        "is_correct": bool(row["correct"]),
        "batch": int(row["batch"]),
    }


def main() -> None:
    df = pd.read_csv(SRC)
    out_path = STAGED / "conversations.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(build_document(row)) + "\n")
    print(f"Wrote {len(df):,} Firestore-shaped documents -> {out_path}")


if __name__ == "__main__":
    main()
