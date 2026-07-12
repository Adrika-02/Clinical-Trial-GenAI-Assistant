"""Shared LangChain tools used by the three agents: SQL execution against
SQLite, clinical note search, and read access to the pre-computed stats /
clustering / classifier artifacts from earlier pipeline steps.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import DEFAULT_DB_PATH

STATS_PATH = PROJECT_ROOT / "data" / "processed" / "stats_results.json"
CLUSTERING_PATH = PROJECT_ROOT / "data" / "processed" / "clustering_results.json"
AE_METRICS_PATH = PROJECT_ROOT / "models" / "saved" / "ae_classifier_metrics.json"

SCHEMA_DESCRIPTION = """\
patients(patient_id, age, gender, bmi, baseline_systolic_bp, baseline_diastolic_bp,
         baseline_hba1c, baseline_ldl, baseline_egfr, treatment_arm, primary_outcome_score,
         dropout_flag, dropout_week, enrollment_date)
visits(visit_id, patient_id, visit_week, visit_date, hba1c, systolic_bp, diastolic_bp,
       ldl_cholesterol, egfr)
adverse_events(ae_id, patient_id, visit_week, ae_code, severity, ae_class)
clinical_notes(note_id, patient_id, visit_week, note_date, note_text, true_ae_class, manually_coded_ae)
patient_clusters(patient_id, cluster, cluster_label)

Notes:
- treatment_arm is 'Drug X' or 'Placebo'.
- severity is 'None'/'Mild'/'Moderate'/'Severe'; ae_class buckets it to 'No AE'/'Mild AE'/'Severe AE'.
- visit_week is one of 0, 2, 4, 8, 12, 24.
- cluster_label is a business name like 'High Responders', 'AE-Prone', 'Non-Responders', 'Stable / Average Responders'.
"""

_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "attach", "pragma", "create")


@tool
def get_database_schema() -> str:
    """Return the SQLite database schema (table and column names) for the clinical trial database."""
    return SCHEMA_DESCRIPTION


@tool
def run_sql_query(sql: str) -> str:
    """Execute a read-only SELECT SQL query against the clinical trial SQLite database and return the results.

    Only SELECT statements are permitted. Always call get_database_schema first if you
    are not certain of the exact table/column names. Results are capped at 50 rows.
    """
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        return "Error: only SELECT statements are permitted."
    if any(word in lowered for word in _FORBIDDEN):
        return "Error: query contains a forbidden keyword."

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    try:
        df = pd.read_sql(sql, conn)
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        conn.close()

    if len(df) == 0:
        return "Query returned no rows."
    truncated = df.head(50)
    return truncated.to_markdown(index=False) + (
        f"\n\n({len(df)} total rows, showing first {len(truncated)})" if len(df) > 50 else ""
    )


@tool
def search_clinical_notes(keyword: str = "", severity: str = "", visit_week: int = -1, limit: int = 15) -> str:
    """Search clinical notes by keyword, AE severity class, and/or visit week.

    Args:
        keyword: substring to search for in note text (case-insensitive), or empty to skip.
        severity: filter to 'No AE', 'Mild AE', or 'Severe AE' (true_ae_class), or empty to skip.
        visit_week: filter to one visit week (0, 2, 4, 8, 12, or 24), or -1 to skip.
        limit: max number of notes to return.
    """
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    query = "SELECT patient_id, visit_week, note_text, true_ae_class FROM clinical_notes WHERE 1=1"
    params = []
    if keyword:
        query += " AND note_text LIKE ?"
        params.append(f"%{keyword}%")
    if severity:
        query += " AND true_ae_class = ?"
        params.append(severity)
    if visit_week >= 0:
        query += " AND visit_week = ?"
        params.append(visit_week)
    query += f" LIMIT {int(limit)}"

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if len(df) == 0:
        return "No matching notes found."
    lines = [
        f"[patient {r.patient_id}, week {r.visit_week}, {r.true_ae_class}] {r.note_text}"
        for r in df.itertuples()
    ]
    return "\n".join(lines)


@tool
def get_stats_summary() -> str:
    """Return the pre-computed statistical test results (Welch's t-test, chi-square,
    Cohen's d, Kruskal-Wallis, confidence intervals) for the trial's primary endpoint and safety comparisons."""
    if not STATS_PATH.exists():
        return "Stats results not yet generated. Run: python -m src.stats.run_analysis"
    with open(STATS_PATH) as f:
        data = json.load(f)
    lines = [data.get("summary", {}).__repr__()]
    for key in ["primary_endpoint_ttest", "hba1c_control_chisq", "ae_occurrence_chisq", "ae_severity_class_chisq", "kruskal_hba1c_by_ae_class"]:
        if key in data:
            lines.append(f"{key}: {data[key]['plain_english']}")
    return "\n".join(lines)


@tool
def get_clustering_summary() -> str:
    """Return the pre-computed K-Means patient cluster profiles (business labels,
    sizes, outcome scores, adverse-event rates, dropout rates)."""
    if not CLUSTERING_PATH.exists():
        return "Clustering results not yet generated. Run: python -m src.clustering.run_clustering"
    with open(CLUSTERING_PATH) as f:
        data = json.load(f)
    return json.dumps(data.get("cluster_profiles", []), indent=2)


@tool
def get_ae_classifier_metrics() -> str:
    """Return the AE classifier's performance metrics (accuracy, precision, recall, F1)
    and the business-impact comparison against manual CRF coding."""
    if not AE_METRICS_PATH.exists():
        return "Classifier metrics not yet generated. Run: python -m src.nlp.train_classifier"
    with open(AE_METRICS_PATH) as f:
        data = json.load(f)
    keep = {k: data[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "business_impact"] if k in data}
    return json.dumps(keep, indent=2)
