"""Generate global + local SHAP explanations for the AE classifier and save
plots/narratives for the README and the Streamlit dashboard.

Run: python -m src.explainability.run_shap_analysis
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
from src.explainability.shap_explain import AEShapExplainer, PLOTS_DIR


def run(db_path: Path = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    notes = pd.read_sql("SELECT note_text, true_ae_class FROM clinical_notes", conn)
    conn.close()

    print("Preprocessing notes for SHAP background/analysis set...")
    notes["clean_text"] = preprocess_batch(notes["note_text"].tolist())

    explainer = AEShapExplainer()
    results = {}

    for class_label in ["Severe AE", "Mild AE"]:
        print(f"Computing global SHAP importance for class: {class_label}")
        subset = notes[notes["true_ae_class"] == class_label]["clean_text"].tolist()
        if len(subset) < 5:
            subset = notes["clean_text"].tolist()
        importance = explainer.global_importance(subset, class_label=class_label, top_k=15)
        plot_path = explainer.plot_global_summary(subset, class_label=class_label)
        results[f"global_{class_label.replace(' ', '_').lower()}"] = {
            "top_features": importance["top_features"],
            "plot_path": str(plot_path.relative_to(PROJECT_ROOT)),
        }
        print(f"  Saved plot to {plot_path}")
        print("  Top features:", [f["feature"] for f in importance["top_features"][:5]])

    # --- Local explanations: one example note per AE class ---
    background = explainer._to_dense(notes["clean_text"].tolist()[:200])
    local_examples = {}
    for class_label in ["No AE", "Mild AE", "Severe AE"]:
        example = notes[notes["true_ae_class"] == class_label].iloc[0]
        narrative = explainer.business_narrative(example["clean_text"], class_label=class_label)
        plot_path = explainer.plot_local_waterfall(
            example["clean_text"], class_label=class_label,
            save_path=PLOTS_DIR / f"local_waterfall_{class_label.replace(' ', '_').lower()}.png",
        )
        local_examples[class_label] = {
            "note_text": example["note_text"],
            "business_narrative": narrative,
            "plot_path": str(plot_path.relative_to(PROJECT_ROOT)),
        }
        print(f"\n[{class_label}] {example['note_text']}")
        print(f"  -> {narrative}")

    results["local_examples"] = local_examples

    out_path = PLOTS_DIR / "shap_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSHAP summary saved to {out_path}")

    return results


if __name__ == "__main__":
    run()
