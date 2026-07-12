"""Page 5 — Upload and Integrate (Interaction 3: Upload).
Mode A: upload a patient CSV -> validate -> append to SQLite -> auto-EDA.
Mode B: upload clinical notes -> NLP pipeline (classify + NER) -> append to SQLite.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from utils import APP_TITLE, page_header, get_connection, clear_all_caches, inject_theme_css

st.set_page_config(page_title=f"Upload — {APP_TITLE}", page_icon="📤", layout="wide")
inject_theme_css()
page_header("📤 Upload and Integrate", "Add new patient records or clinical notes — instantly queryable everywhere else in the app.")

REQUIRED_PATIENT_COLS = [
    "patient_id", "age", "gender", "bmi", "baseline_systolic_bp", "baseline_diastolic_bp",
    "baseline_hba1c", "baseline_ldl", "baseline_egfr", "treatment_arm", "enrollment_date",
]

mode = st.radio("What are you uploading?", ["Patient CSV", "Clinical notes"], horizontal=True)

if mode == "Patient CSV":
    st.write(f"Required columns: `{', '.join(REQUIRED_PATIENT_COLS)}`. Optional: `primary_outcome_score`, `dropout_flag`, `dropout_week`.")
    file = st.file_uploader("Upload patient CSV", type=["csv"])

    if file is not None:
        new_patients = pd.read_csv(file)
        missing = [c for c in REQUIRED_PATIENT_COLS if c not in new_patients.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            conn = get_connection()
            existing_ids = set(pd.read_sql("SELECT patient_id FROM patients", conn)["patient_id"])
            duplicate_ids = existing_ids.intersection(set(new_patients["patient_id"]))

            if duplicate_ids:
                st.warning(f"{len(duplicate_ids)} patient_id(s) already exist and will be skipped: {sorted(duplicate_ids)[:10]}...")
                new_patients = new_patients[~new_patients["patient_id"].isin(duplicate_ids)]

            for col in ["primary_outcome_score", "dropout_week"]:
                if col not in new_patients.columns:
                    new_patients[col] = None
            if "dropout_flag" not in new_patients.columns:
                new_patients["dropout_flag"] = 0

            st.subheader("Auto-EDA Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("New patients to add", len(new_patients))
            c2.metric("Mean age", f"{new_patients['age'].mean():.1f}")
            c3.metric("Arms represented", new_patients["treatment_arm"].nunique())
            st.dataframe(new_patients.describe(include="all"), use_container_width=True)

            fig = px.histogram(new_patients, x="age", color="treatment_arm", nbins=15, title="Age distribution of new upload")
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Integrate into database", type="primary"):
                new_patients.to_sql("patients", conn, if_exists="append", index=False)
                conn.commit()
                clear_all_caches()
                st.success(f"Integrated {len(new_patients)} new patients. They are now queryable in Chat, Cohort Explorer, and the Executive Report.")

else:
    st.write("Upload a CSV with columns `patient_id, visit_week, note_date, note_text` — each note will be run through the NLP pipeline (SpaCy preprocessing, NER, AE classification) and added to the database.")
    file = st.file_uploader("Upload clinical notes CSV", type=["csv"])

    if file is not None:
        new_notes = pd.read_csv(file)
        required = ["patient_id", "visit_week", "note_date", "note_text"]
        missing = [c for c in required if c not in new_notes.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            from src.nlp.analyze_note import analyze_note

            if st.button("Run NLP pipeline and integrate", type="primary"):
                with st.spinner(f"Classifying {len(new_notes)} notes..."):
                    predictions, confidences, severity_scores = [], [], []
                    for text in new_notes["note_text"]:
                        result = analyze_note(str(text))
                        predictions.append(result["prediction"])
                        confidences.append(result["confidence"])
                        severity_scores.append(result["severity_score_0_3"])

                new_notes["true_ae_class"] = predictions
                new_notes["manually_coded_ae"] = predictions  # not yet manually adjudicated; NLP prediction stands in

                st.subheader("Classification Results")
                display_df = new_notes.copy()
                display_df["nlp_confidence"] = confidences
                display_df["severity_score_0_3"] = severity_scores
                st.dataframe(
                    display_df[["patient_id", "visit_week", "note_text", "true_ae_class", "nlp_confidence", "severity_score_0_3"]],
                    use_container_width=True, hide_index=True,
                )

                class_counts = pd.Series(predictions).value_counts()
                fig = px.bar(x=class_counts.index, y=class_counts.values, labels={"x": "AE class", "y": "count"}, title="AE classification breakdown")
                st.plotly_chart(fig, use_container_width=True)

                conn = get_connection()
                new_notes[["patient_id", "visit_week", "note_date", "note_text", "true_ae_class", "manually_coded_ae"]].to_sql(
                    "clinical_notes", conn, if_exists="append", index=False
                )
                conn.commit()
                clear_all_caches()
                st.success(f"Integrated {len(new_notes)} new notes, classified by the NLP pipeline. Now queryable in Chat and Notes Analyser.")
