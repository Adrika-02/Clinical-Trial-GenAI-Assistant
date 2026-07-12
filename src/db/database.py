"""SQLite schema and connection helpers for the clinical trial database."""
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "clinical_trial.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    bmi REAL NOT NULL,
    baseline_systolic_bp REAL NOT NULL,
    baseline_diastolic_bp REAL NOT NULL,
    baseline_hba1c REAL NOT NULL,
    baseline_ldl REAL NOT NULL,
    baseline_egfr REAL NOT NULL,
    treatment_arm TEXT NOT NULL CHECK (treatment_arm IN ('Drug X', 'Placebo')),
    primary_outcome_score REAL,
    dropout_flag INTEGER NOT NULL DEFAULT 0,
    dropout_week INTEGER,
    enrollment_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    visit_week INTEGER NOT NULL,
    visit_date TEXT NOT NULL,
    hba1c REAL,
    systolic_bp REAL,
    diastolic_bp REAL,
    ldl_cholesterol REAL,
    egfr REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS adverse_events (
    ae_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    visit_week INTEGER NOT NULL,
    ae_code TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('None', 'Mild', 'Moderate', 'Severe')),
    ae_class TEXT NOT NULL CHECK (ae_class IN ('No AE', 'Mild AE', 'Severe AE')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    visit_week INTEGER NOT NULL,
    note_date TEXT NOT NULL,
    note_text TEXT NOT NULL,
    true_ae_class TEXT NOT NULL CHECK (true_ae_class IN ('No AE', 'Mild AE', 'Severe AE')),
    manually_coded_ae TEXT NOT NULL CHECK (manually_coded_ae IN ('No AE', 'Mild AE', 'Severe AE')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits(patient_id);
CREATE INDEX IF NOT EXISTS idx_ae_patient ON adverse_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_notes_patient ON clinical_notes(patient_id);
"""


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DEFAULT_DB_PATH, drop_existing: bool = True) -> sqlite3.Connection:
    conn = get_connection(db_path)
    if drop_existing:
        for table in ["clinical_notes", "adverse_events", "visits", "patients"]:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
