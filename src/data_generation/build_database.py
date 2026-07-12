"""End-to-end synthetic data pipeline: generate patients, visits, adverse
events, and clinical notes, then load everything into SQLite (and drop raw
CSVs into data/raw/ for transparency / upload-demo purposes).

Run: python -m src.data_generation.build_database
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.generate_patients import generate_patients
from src.data_generation.generate_visits import simulate_trial
from src.data_generation.generate_notes import generate_notes
from src.db.database import init_db, DEFAULT_DB_PATH

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def build(seed: int = 42):
    print("Generating patient baseline records...")
    patients = generate_patients(seed=seed)

    print("Simulating visits, adverse events, and dropout...")
    patients_out, visits_df, ae_df = simulate_trial(patients, seed=seed)

    print("Generating clinical notes...")
    notes_df = generate_notes(patients_out, visits_df, ae_df, seed=seed + 1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    patients_out.to_csv(RAW_DIR / "patients.csv", index=False)
    visits_df.to_csv(RAW_DIR / "visits.csv", index=False)
    ae_df.to_csv(RAW_DIR / "adverse_events.csv", index=False)
    notes_df.to_csv(RAW_DIR / "clinical_notes.csv", index=False)
    print(f"Raw CSVs written to {RAW_DIR}")

    print("Loading into SQLite...")
    conn = init_db(DEFAULT_DB_PATH, drop_existing=True)
    patients_out.to_sql("patients", conn, if_exists="append", index=False)
    visits_df.to_sql("visits", conn, if_exists="append", index=False)
    ae_df.to_sql("adverse_events", conn, if_exists="append", index=False)
    notes_df.to_sql("clinical_notes", conn, if_exists="append", index=False)
    conn.commit()

    # Sanity-check counts
    summary = {}
    for table in ["patients", "visits", "adverse_events", "clinical_notes"]:
        summary[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()

    print(f"Database written to {DEFAULT_DB_PATH}")
    print("Row counts:", summary)
    return summary


if __name__ == "__main__":
    build()
