"""Run the full statistical analysis suite against the live SQLite database
and save results to data/processed/stats_results.json for reuse by the
Streamlit dashboard and the PDF executive report.

Run: python -m src.stats.run_analysis
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import DEFAULT_DB_PATH
from src.stats.statistical_tests import (
    welch_t_test, chi_square_test, kruskal_wallis_test, normality_check, confidence_interval,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run(db_path: Path = DEFAULT_DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    patients = pd.read_sql("SELECT * FROM patients", conn)
    visits = pd.read_sql("SELECT * FROM visits", conn)
    ae = pd.read_sql("SELECT * FROM adverse_events", conn)
    conn.close()

    results = {}

    # --- Primary endpoint: Welch's t-test on primary_outcome_score ---
    drug_x = patients.loc[patients["treatment_arm"] == "Drug X", "primary_outcome_score"].dropna()
    placebo = patients.loc[patients["treatment_arm"] == "Placebo", "primary_outcome_score"].dropna()

    results["normality_check_outcome_score"] = normality_check(
        patients["primary_outcome_score"].dropna(), metric_name="primary outcome score"
    ).to_dict()

    results["primary_endpoint_ttest"] = welch_t_test(
        drug_x, placebo, label_a="Drug X", label_b="Placebo", metric_name="primary outcome score (0-100)"
    ).to_dict()

    results["ci_outcome_drug_x"] = confidence_interval(drug_x, metric_name="Drug X primary outcome score")
    results["ci_outcome_placebo"] = confidence_interval(placebo, metric_name="Placebo primary outcome score")

    # --- HbA1c <7% at Week 24 (secondary/primary clinical endpoint) ---
    week24 = visits[visits["visit_week"] == 24].merge(patients[["patient_id", "treatment_arm"]], on="patient_id")
    week24["hba1c_controlled"] = week24["hba1c"] < 7.0
    results["hba1c_control_rate_by_arm"] = (
        week24.groupby("treatment_arm")["hba1c_controlled"].mean().round(4).to_dict()
    )
    results["hba1c_control_chisq"] = chi_square_test(
        week24["treatment_arm"], week24["hba1c_controlled"], metric_name="HbA1c <7% control at Week 24"
    ).to_dict()

    # --- Chi-square: AE occurrence rate between arms ---
    ae_with_arm = ae.merge(patients[["patient_id", "treatment_arm"]], on="patient_id")
    ae_with_arm["ae_occurred"] = ae_with_arm["ae_code"] != "None"
    # one row per patient: did they ever have an AE
    ae_ever = ae_with_arm.groupby("patient_id").agg(
        treatment_arm=("treatment_arm", "first"), ae_occurred=("ae_occurred", "max")
    ).reset_index()
    results["ae_occurrence_chisq"] = chi_square_test(
        ae_ever["treatment_arm"], ae_ever["ae_occurred"], metric_name="adverse event occurrence"
    ).to_dict()

    # --- Chi-square: AE severity class distribution between arms ---
    results["ae_severity_class_chisq"] = chi_square_test(
        ae_with_arm["treatment_arm"], ae_with_arm["ae_class"], metric_name="adverse event severity class"
    ).to_dict()

    # --- Kruskal-Wallis: HbA1c reduction (baseline - week24) across AE severity classes ---
    patient_max_severity = ae_with_arm.groupby("patient_id")["ae_class"].agg(
        lambda s: s.map({"No AE": 0, "Mild AE": 1, "Severe AE": 2}).max()
    )
    severity_map_rev = {0: "No AE", 1: "Mild AE", 2: "Severe AE"}
    patients["max_ae_class"] = patients["patient_id"].map(patient_max_severity).map(severity_map_rev)

    week24_hba1c = week24.set_index("patient_id")["hba1c"]
    patients_reduction = patients.set_index("patient_id")
    patients_reduction["hba1c_reduction"] = patients_reduction["baseline_hba1c"] - week24_hba1c

    groups, names = [], []
    for cls in ["No AE", "Mild AE", "Severe AE"]:
        vals = patients_reduction.loc[patients_reduction["max_ae_class"] == cls, "hba1c_reduction"].dropna()
        if len(vals) > 0:
            groups.append(vals.values)
            names.append(cls)

    results["kruskal_hba1c_by_ae_class"] = kruskal_wallis_test(
        *groups, group_names=names, metric_name="HbA1c reduction by Week 24, grouped by worst AE severity"
    ).to_dict()

    # --- Business-facing summary numbers ---
    results["summary"] = {
        "n_patients": int(len(patients)),
        "n_drug_x": int((patients["treatment_arm"] == "Drug X").sum()),
        "n_placebo": int((patients["treatment_arm"] == "Placebo").sum()),
        "completion_rate": round(float((patients["dropout_flag"] == 0).mean()), 4),
        "overall_ae_rate": round(float(ae_ever["ae_occurred"].mean()), 4),
        "ae_rate_drug_x": round(float(ae_ever.loc[ae_ever.treatment_arm == "Drug X", "ae_occurred"].mean()), 4),
        "ae_rate_placebo": round(float(ae_ever.loc[ae_ever.treatment_arm == "Placebo", "ae_occurred"].mean()), 4),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "stats_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {out_path}\n")
    for key in ["primary_endpoint_ttest", "hba1c_control_chisq", "ae_occurrence_chisq", "ae_severity_class_chisq", "kruskal_hba1c_by_ae_class"]:
        print(f"--- {key} ---")
        print(results[key]["plain_english"])
        print()

    return results


if __name__ == "__main__":
    run()
