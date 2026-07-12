"""Generate patient-level baseline records for the synthetic clinical trial.

Trial context: Drug X is a synthetic anti-diabetic agent being compared to
placebo. Baseline vitals and labs are drawn from clinically plausible ranges
for a Type 2 diabetes population enrolling in a Phase III trial.
"""
import numpy as np
import pandas as pd

N_PATIENTS = 500
N_PER_ARM = N_PATIENTS // 2
VISIT_WEEKS = [0, 2, 4, 8, 12, 24]
ENROLLMENT_START = pd.Timestamp("2024-01-08")


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_patients(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    patient_id = np.arange(1, N_PATIENTS + 1)
    treatment_arm = np.array(["Drug X"] * N_PER_ARM + ["Placebo"] * N_PER_ARM)
    rng.shuffle(treatment_arm)

    age = _clip(rng.normal(55, 10, N_PATIENTS), 30, 75).round().astype(int)
    gender = rng.choice(["Female", "Male"], size=N_PATIENTS, p=[0.52, 0.48])
    bmi = _clip(rng.normal(30, 5, N_PATIENTS), 18.5, 45).round(1)

    baseline_systolic_bp = _clip(rng.normal(136, 14, N_PATIENTS), 100, 180).round(1)
    baseline_diastolic_bp = _clip(rng.normal(85, 9, N_PATIENTS), 60, 110).round(1)
    baseline_hba1c = _clip(rng.normal(8.1, 1.0, N_PATIENTS), 6.0, 12.0).round(2)
    baseline_ldl = _clip(rng.normal(128, 26, N_PATIENTS), 70, 220).round(1)
    baseline_egfr = _clip(rng.normal(85, 15, N_PATIENTS), 30, 120).round(1)

    enrollment_offsets = rng.integers(0, 180, size=N_PATIENTS)
    enrollment_date = [
        (ENROLLMENT_START + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in enrollment_offsets
    ]

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "baseline_systolic_bp": baseline_systolic_bp,
            "baseline_diastolic_bp": baseline_diastolic_bp,
            "baseline_hba1c": baseline_hba1c,
            "baseline_ldl": baseline_ldl,
            "baseline_egfr": baseline_egfr,
            "treatment_arm": treatment_arm,
            "enrollment_date": enrollment_date,
        }
    )
    return df


if __name__ == "__main__":
    patients = generate_patients()
    print(patients.head())
    print(patients["treatment_arm"].value_counts())
