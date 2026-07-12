"""Simulate longitudinal visit data, adverse events, dropout, and the primary
outcome score for each patient across the 6 scheduled visits.

The treatment effect is modelled as a saturating dose-response curve with
patient-level heterogeneity (a random "responsiveness" draw per patient) so
that downstream clustering has a genuine latent structure to recover, and
downstream hypothesis tests have a real, non-trivial effect to detect.
"""
import numpy as np
import pandas as pd

VISIT_WEEKS = [0, 2, 4, 8, 12, 24]

AE_CODES = ["Nausea", "Dizziness", "Elevated BP", "Chest Pain", "Fatigue"]

SEVERITY_GIVEN_AE = {
    "Drug X": {"Mild": 0.55, "Moderate": 0.32, "Severe": 0.13},
    "Placebo": {"Mild": 0.72, "Moderate": 0.23, "Severe": 0.05},
}

AE_CODE_GIVEN_SEVERITY = {
    "Mild": {"Nausea": 0.35, "Fatigue": 0.30, "Dizziness": 0.20, "Elevated BP": 0.10, "Chest Pain": 0.05},
    "Moderate": {"Dizziness": 0.30, "Elevated BP": 0.30, "Nausea": 0.20, "Fatigue": 0.15, "Chest Pain": 0.05},
    "Severe": {"Chest Pain": 0.35, "Elevated BP": 0.30, "Dizziness": 0.20, "Nausea": 0.10, "Fatigue": 0.05},
}

SEVERITY_TO_CLASS = {"None": "No AE", "Mild": "Mild AE", "Moderate": "Mild AE", "Severe": "Severe AE"}

# Probability an AE occurs at a given visit, by arm and visit week (titration
# period in early weeks carries a higher side-effect burden).
AE_PROB_BY_WEEK = {
    "Drug X": {0: 0.05, 2: 0.32, 4: 0.30, 8: 0.22, 12: 0.18, 24: 0.14},
    "Placebo": {0: 0.03, 2: 0.14, 4: 0.13, 8: 0.12, 12: 0.11, 24: 0.10},
}


def _sample_choice(rng, options_probs):
    options = list(options_probs.keys())
    probs = np.array(list(options_probs.values()))
    probs = probs / probs.sum()
    return options[rng.choice(len(options), p=probs)]


def simulate_trial(patients: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(patients)

    # Per-patient latent responsiveness (heterogeneity used later for clustering)
    hba1c_max_effect = np.where(
        patients["treatment_arm"] == "Drug X",
        np.clip(rng.normal(1.3, 0.45, n), 0.2, 2.6),
        np.clip(rng.normal(0.3, 0.2, n), 0.0, 1.0),
    )
    sbp_max_effect = np.where(
        patients["treatment_arm"] == "Drug X",
        np.clip(rng.normal(7.0, 3.0, n), 0, 16),
        np.clip(rng.normal(1.0, 2.0, n), -3, 6),
    )
    ldl_max_effect = np.where(
        patients["treatment_arm"] == "Drug X",
        np.clip(rng.normal(13.0, 5.0, n), 0, 30),
        np.clip(rng.normal(2.0, 3.0, n), -5, 10),
    )
    egfr_decline_rate = np.where(
        patients["treatment_arm"] == "Drug X",
        np.clip(rng.normal(0.05, 0.03, n), 0, 0.15),
        np.clip(rng.normal(0.09, 0.04, n), 0, 0.2),
    )

    visit_rows = []
    ae_rows = []
    dropout_flag = np.zeros(n, dtype=int)
    dropout_week = np.full(n, np.nan)
    max_severity_rank = np.zeros(n, dtype=int)
    severity_rank = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

    last_labs = {}

    for i, patient in patients.reset_index(drop=True).iterrows():
        pid = patient["patient_id"]
        arm = patient["treatment_arm"]
        enroll_date = pd.Timestamp(patient["enrollment_date"])
        has_dropped = False
        drop_week_val = None
        last_valid = {
            "hba1c": patient["baseline_hba1c"],
            "systolic_bp": patient["baseline_systolic_bp"],
            "diastolic_bp": patient["baseline_diastolic_bp"],
            "ldl_cholesterol": patient["baseline_ldl"],
            "egfr": patient["baseline_egfr"],
        }

        for week in VISIT_WEEKS:
            visit_date = (enroll_date + pd.Timedelta(days=week * 7)).strftime("%Y-%m-%d")

            if not has_dropped:
                hba1c_improve = hba1c_max_effect[i] * (1 - np.exp(-week / 8))
                sbp_improve = sbp_max_effect[i] * (1 - np.exp(-week / 8))
                ldl_improve = ldl_max_effect[i] * (1 - np.exp(-week / 10))

                hba1c = patient["baseline_hba1c"] - hba1c_improve + rng.normal(0, 0.15)
                systolic_bp = patient["baseline_systolic_bp"] - sbp_improve + rng.normal(0, 4)
                diastolic_bp = patient["baseline_diastolic_bp"] - 0.6 * sbp_improve + rng.normal(0, 3)
                ldl_cholesterol = patient["baseline_ldl"] - ldl_improve + rng.normal(0, 6)
                egfr = patient["baseline_egfr"] - egfr_decline_rate[i] * week + rng.normal(0, 3)

                hba1c = float(np.clip(hba1c, 5.0, 13.0))
                systolic_bp = float(np.clip(systolic_bp, 90, 190))
                diastolic_bp = float(np.clip(diastolic_bp, 55, 115))
                ldl_cholesterol = float(np.clip(ldl_cholesterol, 50, 230))
                egfr = float(np.clip(egfr, 15, 125))

                last_valid = {
                    "hba1c": hba1c, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
                    "ldl_cholesterol": ldl_cholesterol, "egfr": egfr,
                }
            else:
                # Post-dropout: last-observation-carried-forward with mild drift
                hba1c = float(np.clip(last_valid["hba1c"] + rng.normal(0.05, 0.1), 5.0, 13.0))
                systolic_bp = float(np.clip(last_valid["systolic_bp"] + rng.normal(0.3, 3), 90, 190))
                diastolic_bp = float(np.clip(last_valid["diastolic_bp"] + rng.normal(0.2, 2), 55, 115))
                ldl_cholesterol = float(np.clip(last_valid["ldl_cholesterol"] + rng.normal(0.5, 4), 50, 230))
                egfr = float(np.clip(last_valid["egfr"] - rng.uniform(0, 0.5), 15, 125))
                last_valid = {
                    "hba1c": hba1c, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
                    "ldl_cholesterol": ldl_cholesterol, "egfr": egfr,
                }

            visit_rows.append({
                "patient_id": pid, "visit_week": week, "visit_date": visit_date,
                "hba1c": round(hba1c, 2), "systolic_bp": round(systolic_bp, 1),
                "diastolic_bp": round(diastolic_bp, 1), "ldl_cholesterol": round(ldl_cholesterol, 1),
                "egfr": round(egfr, 1),
            })

            # Adverse event assignment (none once patient has dropped out / stopped active monitoring)
            if has_dropped:
                ae_code, severity = "None", "None"
            else:
                ae_prob = AE_PROB_BY_WEEK[arm][week]
                if rng.random() < ae_prob:
                    severity = _sample_choice(rng, SEVERITY_GIVEN_AE[arm])
                    ae_code = _sample_choice(rng, AE_CODE_GIVEN_SEVERITY[severity])
                else:
                    ae_code, severity = "None", "None"

            ae_rows.append({
                "patient_id": pid, "visit_week": week, "ae_code": ae_code,
                "severity": severity, "ae_class": SEVERITY_TO_CLASS[severity],
            })

            if severity_rank[severity] > max_severity_rank[i]:
                max_severity_rank[i] = severity_rank[severity]

            # Dropout decision after recording this visit's AE
            if not has_dropped:
                drop_prob = 0.015  # background lost-to-follow-up
                if severity == "Severe":
                    drop_prob += 0.55
                elif severity == "Moderate":
                    drop_prob += 0.08
                if rng.random() < drop_prob and week != VISIT_WEEKS[-1]:
                    has_dropped = True
                    drop_week_val = week

        if has_dropped:
            dropout_flag[i] = 1
            dropout_week[i] = drop_week_val

    visits_df = pd.DataFrame(visit_rows)
    ae_df = pd.DataFrame(ae_rows)

    # Primary outcome score at week 24 (composite, 0-100)
    week24 = visits_df[visits_df["visit_week"] == 24].set_index("patient_id")
    patients_idx = patients.set_index("patient_id")

    severity_penalty = np.select(
        [max_severity_rank == 3, max_severity_rank == 2, max_severity_rank == 1],
        [15, 6, 2],
        default=0,
    )

    hba1c_delta = patients_idx["baseline_hba1c"].values - week24.loc[patients_idx.index, "hba1c"].values
    sbp_delta = patients_idx["baseline_systolic_bp"].values - week24.loc[patients_idx.index, "systolic_bp"].values
    ldl_delta = patients_idx["baseline_ldl"].values - week24.loc[patients_idx.index, "ldl_cholesterol"].values

    noise = rng.normal(0, 8, n)
    score = 50 + 10 * hba1c_delta + 0.5 * sbp_delta + 0.15 * ldl_delta - severity_penalty + noise
    primary_outcome_score = np.clip(score, 0, 100).round(1)

    patients_out = patients.copy()
    patients_out["primary_outcome_score"] = primary_outcome_score
    patients_out["dropout_flag"] = dropout_flag
    patients_out["dropout_week"] = dropout_week

    return patients_out, visits_df, ae_df


if __name__ == "__main__":
    from generate_patients import generate_patients

    patients = generate_patients()
    patients_out, visits_df, ae_df = simulate_trial(patients)
    print(patients_out[["treatment_arm", "primary_outcome_score", "dropout_flag"]].groupby("treatment_arm").mean())
    print(visits_df.shape, ae_df.shape)
    print(ae_df["ae_class"].value_counts())
