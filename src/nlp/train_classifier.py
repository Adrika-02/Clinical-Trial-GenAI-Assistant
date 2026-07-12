"""Train and evaluate the TF-IDF + Logistic Regression adverse-event
classifier against the clinical notes in SQLite, then quantify how many
more adverse events the NLP model recovers versus the historical manual
CRF-coding baseline.

Run: python -m src.nlp.train_classifier
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import DEFAULT_DB_PATH
from src.nlp.preprocessing import preprocess_batch
from src.nlp.ae_classifier import (
    build_pipeline, train_test_split_stratified, evaluate, save_model, CLASSES,
)

MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "ae_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "saved" / "ae_classifier_metrics.json"


def run(db_path: Path = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    notes = pd.read_sql("SELECT note_text, true_ae_class, manually_coded_ae FROM clinical_notes", conn)
    conn.close()

    print(f"Loaded {len(notes)} clinical notes. Class distribution (true label):")
    print(notes["true_ae_class"].value_counts())

    print("Running spaCy preprocessing (tokenize, lemmatize, remove stopwords)...")
    notes["clean_text"] = preprocess_batch(notes["note_text"].tolist())

    X = notes["clean_text"]
    y = notes["true_ae_class"]
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    manual_test = notes.loc[X_test.index, "manually_coded_ae"]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    print("\n=== Model performance (held-out test set) ===")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):     {metrics['f1_weighted']:.4f}")

    # --- Business metric: NLP-detected vs manually-coded AE uplift (test set only) ---
    y_pred = pipeline.predict(X_test)
    y_test_arr = y_test.values
    manual_arr = manual_test.values

    manual_ae_count = int((manual_arr != "No AE").sum())
    nlp_ae_count = int((y_pred != "No AE").sum())
    uplift_pct = round((nlp_ae_count - manual_ae_count) / manual_ae_count * 100, 1) if manual_ae_count > 0 else None

    missed_mask = (manual_arr == "No AE") & (y_test_arr != "No AE")  # true AE that manual coding missed
    n_missed = int(missed_mask.sum())
    n_missed_recovered = int(((y_pred != "No AE") & missed_mask).sum())
    recovery_rate = round(n_missed_recovered / n_missed * 100, 1) if n_missed > 0 else None

    business_metrics = {
        "test_set_size": int(len(y_test)),
        "manual_coded_ae_count": manual_ae_count,
        "nlp_detected_ae_count": nlp_ae_count,
        "uplift_pct": uplift_pct,
        "manually_missed_ae_count": n_missed,
        "nlp_recovered_of_missed": n_missed_recovered,
        "recovery_rate_pct_of_missed": recovery_rate,
    }
    print("\n=== NLP vs. manual coding (test set) ===")
    print(json.dumps(business_metrics, indent=2))

    metrics["business_impact"] = business_metrics
    metrics["train_size"] = int(len(X_train))
    metrics["test_size"] = int(len(X_test))
    metrics["classes"] = CLASSES

    save_model(pipeline, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    return pipeline, metrics


if __name__ == "__main__":
    run()
