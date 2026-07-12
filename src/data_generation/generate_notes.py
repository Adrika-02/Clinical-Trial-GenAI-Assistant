"""Generate realistic free-text clinical notes for every patient-visit,
grounded in that visit's actual lab values and adverse-event assignment so
the NLP pipeline can be validated against genuine structured ground truth.
"""
import numpy as np
import pandas as pd

SYMPTOM_TEXT = {
    "Nausea": "nausea",
    "Dizziness": "dizziness",
    "Elevated BP": "elevated blood pressure",
    "Chest Pain": "chest pain",
    "Fatigue": "fatigue",
}

NO_AE_TEMPLATES = [
    "Week {week} visit: Patient tolerating treatment well. {lab_sentence} No adverse events reported this cycle.",
    "Patient reports feeling well since last visit. {lab_sentence} No new complaints. Continue current dose.",
    "Routine follow-up visit. {lab_sentence} No adverse events noted. Medication adherence confirmed.",
    "Patient in good spirits, denies any new symptoms. {lab_sentence} Plan: continue on current regimen.",
]

MILD_TEMPLATES = [
    "Patient reports mild {symptom} following Day {day} dose. {vital_sentence} No changes to medication recommended.",
    "Mild {symptom} noted at this visit. Symptom resolved without intervention. {lab_sentence}",
    "Patient describes transient, mild {symptom} over the past week. {vital_sentence} Advised to monitor and report if worsening.",
]

MODERATE_TEMPLATES = [
    "Moderate {symptom} reported. Dose reduced from {dose_from}mg to {dose_to}mg. Follow-up scheduled in 2 weeks.",
    "Patient experiencing moderate {symptom}. {vital_sentence} Monitoring closely; dose adjustment considered.",
    "Moderate {symptom} since last visit, impacting daily activities. {vital_sentence} Dose adjusted, close follow-up planned.",
]

SEVERE_TEMPLATES = [
    "Severe {symptom} reported at Week {week} visit. {vital_sentence} Patient referred for further evaluation. Treatment temporarily suspended.",
    "Significant {symptom} with {vital_sentence} Considering discontinuation of study drug. Urgent clinical review ordered.",
    "Patient presented with severe {symptom}. {vital_sentence} Study drug held pending safety assessment.",
]

DISCONTINUED_TEMPLATES = [
    "Week {week} safety follow-up: Patient discontinued study drug at Week {dropout_week} due to prior adverse event. {lab_sentence} No new complaints at this visit.",
    "Safety follow-up visit only; patient remains off study drug since Week {dropout_week}. {lab_sentence} Vitals stable.",
]


def _lab_sentence(rng, hba1c, prev_hba1c, sbp, dbp):
    if prev_hba1c is not None and round(prev_hba1c, 1) > round(hba1c, 1):
        hba1c_phrase = f"HbA1c improved from {round(prev_hba1c,1)} to {round(hba1c,1)}."
    elif prev_hba1c is not None and round(prev_hba1c, 1) < round(hba1c, 1):
        hba1c_phrase = f"HbA1c rose slightly from {round(prev_hba1c,1)} to {round(hba1c,1)}."
    else:
        hba1c_phrase = f"HbA1c stable at {round(hba1c,1)}."

    variants = [
        hba1c_phrase,
        f"Blood pressure stable at {int(round(sbp))}/{int(round(dbp))}.",
        f"Vitals within expected range; HbA1c {round(hba1c,1)}, BP {int(round(sbp))}/{int(round(dbp))}.",
    ]
    return rng.choice(variants)


def _vital_sentence(rng, sbp, dbp):
    variants = [
        f"BP slightly elevated at {int(round(sbp))}/{int(round(dbp))}.",
        f"Vitals recorded: BP {int(round(sbp))}/{int(round(dbp))}.",
        f"BP measured at {int(round(sbp))}/{int(round(dbp))} at this visit.",
    ]
    return rng.choice(variants)


def generate_notes(patients: pd.DataFrame, visits_df: pd.DataFrame, ae_df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patients_idx = patients.set_index("patient_id")
    visits_sorted = visits_df.sort_values(["patient_id", "visit_week"])
    ae_lookup = ae_df.set_index(["patient_id", "visit_week"])

    notes = []
    prev_hba1c_by_patient = {}

    for _, visit in visits_sorted.iterrows():
        pid = visit["patient_id"]
        week = visit["visit_week"]
        patient = patients_idx.loc[pid]
        ae_row = ae_lookup.loc[(pid, week)]
        severity = ae_row["severity"]
        ae_code = ae_row["ae_code"]
        ae_class = ae_row["ae_class"]

        dropped_before_this_visit = (
            patient["dropout_flag"] == 1 and pd.notna(patient["dropout_week"]) and week > patient["dropout_week"]
        )

        prev_hba1c = prev_hba1c_by_patient.get(pid)

        if dropped_before_this_visit:
            template = rng.choice(DISCONTINUED_TEMPLATES)
            text = template.format(
                week=week,
                dropout_week=int(patient["dropout_week"]),
                lab_sentence=_lab_sentence(rng, visit["hba1c"], prev_hba1c, visit["systolic_bp"], visit["diastolic_bp"]),
            )
        elif severity == "None":
            template = rng.choice(NO_AE_TEMPLATES)
            text = template.format(
                week=week,
                lab_sentence=_lab_sentence(rng, visit["hba1c"], prev_hba1c, visit["systolic_bp"], visit["diastolic_bp"]),
            )
        else:
            symptom = SYMPTOM_TEXT[ae_code]
            day = int(week * 7 - rng.integers(0, 6)) if week > 0 else rng.integers(1, 5)
            vital_sentence = _vital_sentence(rng, visit["systolic_bp"], visit["diastolic_bp"])
            lab_sentence = _lab_sentence(rng, visit["hba1c"], prev_hba1c, visit["systolic_bp"], visit["diastolic_bp"])

            if severity == "Mild":
                template = rng.choice(MILD_TEMPLATES)
                text = template.format(symptom=symptom, day=max(day, 1), vital_sentence=vital_sentence, lab_sentence=lab_sentence)
            elif severity == "Moderate":
                template = rng.choice(MODERATE_TEMPLATES)
                dose_from = int(rng.choice([20, 40]))
                dose_to = dose_from // 2
                text = template.format(symptom=symptom, dose_from=dose_from, dose_to=dose_to, vital_sentence=vital_sentence)
            else:  # Severe
                template = rng.choice(SEVERE_TEMPLATES)
                text = template.format(week=week, symptom=symptom, vital_sentence=vital_sentence)

        prev_hba1c_by_patient[pid] = visit["hba1c"]

        notes.append({
            "patient_id": pid,
            "visit_week": week,
            "note_date": visit["visit_date"],
            "note_text": text,
            "manually_coded_ae": ae_class,
        })

    return pd.DataFrame(notes)


if __name__ == "__main__":
    from generate_patients import generate_patients
    from generate_visits import simulate_trial

    patients = generate_patients()
    patients_out, visits_df, ae_df = simulate_trial(patients)
    notes_df = generate_notes(patients_out, visits_df, ae_df)
    print(notes_df.shape)
    print(notes_df.sample(5, random_state=1)[["visit_week", "note_text", "manually_coded_ae"]].to_string())
