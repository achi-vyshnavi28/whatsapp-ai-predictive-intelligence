"""
Load step. Run this yourself, after setting GOOGLE_APPLICATION_CREDENTIALS
to point at your own Firebase service-account JSON key -- this script
never asks for or hardcodes credentials.

Usage:
    python build_firestore_documents.py   # stage locally (no credentials)
    python load_to_firestore.py            # load (needs your own Firebase project)
"""
import json
import os
from pathlib import Path

from google.cloud import firestore

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / "data" / "staged" / "conversations.jsonl"
COLLECTION_NAME = "conversations"


def main() -> None:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise SystemExit(
            "Missing required environment variable GOOGLE_APPLICATION_CREDENTIALS. "
            "Set it to the path of your own Firebase service-account JSON key before running this script, "
            "e.g.\n  set GOOGLE_APPLICATION_CREDENTIALS=C:\\path\\to\\serviceAccountKey.json"
        )

    db = firestore.Client()
    batch = db.batch()
    n = 0
    with STAGED.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            ref = db.collection(COLLECTION_NAME).document(doc["conversation_id"])
            batch.set(ref, doc)
            n += 1
            if n % 400 == 0:
                batch.commit()
                batch = db.batch()
    batch.commit()
    print(f"Loaded {n:,} documents into Firestore collection '{COLLECTION_NAME}'")

    print("\n--- Example Firestore queries ---")
    low_conf = db.collection(COLLECTION_NAME).where("confidence", "<", 0.5).limit(5).stream()
    print("Low-confidence predictions (confidence < 0.5):")
    for doc in low_conf:
        d = doc.to_dict()
        print(f"  {d['conversation_id']}: '{d['utterance'][:50]}...' -> {d['predicted_intent']} ({d['confidence']})")

    incorrect = list(db.collection(COLLECTION_NAME).where("is_correct", "==", False).stream())
    print(f"\nMisclassified documents: {len(incorrect)}")


if __name__ == "__main__":
    main()
