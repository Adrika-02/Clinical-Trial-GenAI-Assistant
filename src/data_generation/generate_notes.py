"""Generate realistic free-text clinical notes for every patient-visit,
grounded in that visit's actual lab values and adverse-event assignment so
the NLP pipeline can be validated against genuine structured ground truth.

Lexical variety (synonyms, paraphrasing without explicit severity words,
and unrelated distractor sentences in "no event" notes) is deliberately
injected so the downstream text classifier has to learn genuine language
patterns rather than a single deterministic keyword-to-label mapping.
"""
import numpy as np
import pandas as pd

SYMPTOM_SYNONYMS = {
    "Nausea": ["nausea", "queasiness", "an upset stomach", "feeling sick to the stomach"],
    "Dizziness": ["dizziness", "light-headedness", "feeling unsteady on their feet", "a spinning sensation"],
    "Elevated BP": ["elevated blood pressure", "blood pressure trending higher than baseline", "a rise in blood pressure readings"],
    "Chest Pain": ["chest pain", "chest discomfort", "tightness in the chest", "pressure in the chest"],
    "Fatigue": ["fatigue", "tiredness", "low energy", "a general sense of exhaustion"],
}

NO_AE_TEMPLATES = [
    "Week {week} visit: Patient tolerating treatment well. {lab_sentence} No adverse events reported this cycle.",
    "Patient reports feeling well since last visit. {lab_sentence} No new complaints. Continue current dose.",
    "Routine follow-up visit. {lab_sentence} No adverse events noted. Medication adherence confirmed.",
    "Patient in good spirits, denies any new symptoms. {lab_sentence} Plan: continue on current regimen.",
]

DISTRACTOR_SENTENCES = [
    "Patient mentions a mild seasonal allergy flare, unrelated to study medication.",
    "History of intermittent tension headaches predates enrollment; no change this cycle.",
    "Patient notes occasional mild knee stiffness from an old injury, not felt to be treatment related.",
    "Brief mention of work-related stress; no physical symptoms or adverse events reported.",
    "Patient recovering from a minor cold last week, considered unrelated to study drug.",
]

MILD_TEMPLATES = [
    "Patient reports mild {symptom} following Day {day} dose. {vital_sentence} No changes to medication recommended.",
    "Mild {symptom} noted at this visit. Symptom resolved without intervention. {lab_sentence}",
    "Patient describes transient, mild {symptom} over the past week. {vital_sentence} Advised to monitor and report if worsening.",
    "Patient experienced a brief episode of {symptom} that resolved on its own within a day. {lab_sentence}",
    "{symptom_cap} described by the patient as minor and short-lived, not affecting daily activities. {vital_sentence}",
]

MODERATE_TEMPLATES = [
    "Moderate {symptom} reported. Dose reduced from {dose_from}mg to {dose_to}mg. Follow-up scheduled in 2 weeks.",
    "Patient experiencing moderate {symptom}. {vital_sentence} Monitoring closely; dose adjustment considered.",
    "Moderate {symptom} since last visit, impacting daily activities. {vital_sentence} Dose adjusted, close follow-up planned.",
    "{symptom_cap} affecting the patient's daily routine this week, difficult to ignore. Dose reduced from {dose_from}mg to {dose_to}mg.",
    "Patient reports persistent {symptom} prompting a dose change at this visit. {vital_sentence}",
]

SEVERE_TEMPLATES = [
    "Severe {symptom} reported at Week {week} visit. {vital_sentence} Patient referred for further evaluation. Treatment temporarily suspended.",
    "Significant {symptom} with {vital_sentence} Considering discontinuation of study drug. Urgent clinical review ordered.",
    "Patient presented with severe {symptom}. {vital_sentence} Study drug held pending safety assessment.",
    "{symptom_cap} serious enough to warrant an urgent evaluation this visit. Study drug held pending safety assessment.",
    "The care team considered the patient's {symptom} clinically concerning; treatment was suspended pending review. {vital_sentence}",
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


def _apply_manual_coding_noise(rng, true_class: str) -> str:
    """Simulate imperfect manual CRF (case report form) coding: investigators
    write the true symptom in the free-text note, but under-report it in the
    structured adverse-event case report form at realistic real-world rates.
    This creates a genuine, non-circular basis for measuring how much more
    the NLP classifier (trained against the adjudicated ground truth) can
    recover versus the historical manual-coding baseline.
    """
    if true_class == "No AE":
        return "No AE"
    if true_class == "Mild AE":
        return "No AE" if rng.random() < 0.25 else "Mild AE"
    # Severe AE
    r = rng.random()
    if r < 0.02:
        return "No AE"
    if r < 0.10:
        return "Mild AE"
    return "Severe AE"


_SEVERITY_ORDER = ["None", "Mild", "Moderate", "Severe"]


def _perturb_text_severity(rng, severity: str, p: float = 0.15) -> str:
    """Structured severity grading (protocol-driven, e.g. vital-sign thresholds
    or CTCAE criteria) does not always match the qualitative language an
    investigator happens to use in the free-text note. With probability p,
    the note is written one grade off from the adjudicated structured
    severity — the same real-world ambiguity that makes AE detection from
    free text a genuinely hard NLP problem rather than a keyword lookup.
    """
    if rng.random() >= p:
        return severity
    idx = _SEVERITY_ORDER.index(severity)
    shift = rng.choice([-1, 1])
    new_idx = min(max(idx + shift, 0), len(_SEVERITY_ORDER) - 1)
    return _SEVERITY_ORDER[new_idx]


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
        else:
            text_severity = _perturb_text_severity(rng, severity)

            if text_severity == "None":
                template = rng.choice(NO_AE_TEMPLATES)
                text = template.format(
                    week=week,
                    lab_sentence=_lab_sentence(rng, visit["hba1c"], prev_hba1c, visit["systolic_bp"], visit["diastolic_bp"]),
                )
                if rng.random() < 0.25:
                    text += " " + rng.choice(DISTRACTOR_SENTENCES)
            else:
                effective_ae_code = ae_code if ae_code != "None" else rng.choice(list(SYMPTOM_SYNONYMS.keys()))
                symptom = rng.choice(SYMPTOM_SYNONYMS[effective_ae_code])
                symptom_cap = symptom[0].upper() + symptom[1:]
                day = int(week * 7 - rng.integers(0, 6)) if week > 0 else rng.integers(1, 5)
                vital_sentence = _vital_sentence(rng, visit["systolic_bp"], visit["diastolic_bp"])
                lab_sentence = _lab_sentence(rng, visit["hba1c"], prev_hba1c, visit["systolic_bp"], visit["diastolic_bp"])

                if text_severity == "Mild":
                    template = rng.choice(MILD_TEMPLATES)
                    text = template.format(
                        symptom=symptom, symptom_cap=symptom_cap, day=max(day, 1),
                        vital_sentence=vital_sentence, lab_sentence=lab_sentence,
                    )
                elif text_severity == "Moderate":
                    template = rng.choice(MODERATE_TEMPLATES)
                    dose_from = int(rng.choice([20, 40]))
                    dose_to = dose_from // 2
                    text = template.format(
                        symptom=symptom, symptom_cap=symptom_cap, dose_from=dose_from,
                        dose_to=dose_to, vital_sentence=vital_sentence,
                    )
                else:  # Severe
                    template = rng.choice(SEVERE_TEMPLATES)
                    text = template.format(week=week, symptom=symptom, symptom_cap=symptom_cap, vital_sentence=vital_sentence)

        prev_hba1c_by_patient[pid] = visit["hba1c"]

        notes.append({
            "patient_id": pid,
            "visit_week": week,
            "note_date": visit["visit_date"],
            "note_text": text,
            "true_ae_class": ae_class,
            "manually_coded_ae": _apply_manual_coding_noise(rng, ae_class),
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
