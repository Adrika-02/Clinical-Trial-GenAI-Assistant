import numpy as np
import pandas as pd

from src.clustering.patient_clustering import (
    build_feature_table, scale_features, fit_kmeans, pca_projection,
    profile_clusters, label_all_clusters,
)


def _toy_patients():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "patient_id": range(1, n + 1),
        "age": rng.integers(30, 75, n),
        "bmi": rng.normal(30, 5, n),
        "baseline_systolic_bp": rng.normal(135, 10, n),
        "baseline_diastolic_bp": rng.normal(85, 8, n),
        "baseline_hba1c": rng.normal(8, 1, n),
        "baseline_ldl": rng.normal(130, 20, n),
        "baseline_egfr": rng.normal(85, 10, n),
        "treatment_arm": ["Drug X"] * 30 + ["Placebo"] * 30,
        "primary_outcome_score": np.concatenate([rng.normal(70, 5, 30), rng.normal(40, 5, 30)]),
        "dropout_flag": rng.integers(0, 2, n),
    })


def _toy_ae(patients):
    rows = []
    for pid in patients["patient_id"]:
        severity = "Severe" if pid % 10 == 0 else "None"
        rows.append({"patient_id": pid, "visit_week": 0, "ae_code": "Nausea", "severity": severity})
    return pd.DataFrame(rows)


def test_build_feature_table_fills_missing_ae_with_zero():
    patients = _toy_patients()
    ae = _toy_ae(patients)
    df = build_feature_table(patients, ae)
    assert "max_ae_severity_rank" in df.columns
    assert df["max_ae_severity_rank"].isna().sum() == 0


def test_kmeans_produces_expected_cluster_count_and_valid_silhouette():
    patients = _toy_patients()
    ae = _toy_ae(patients)
    df = build_feature_table(patients, ae)
    X, _ = scale_features(df)
    km, labels, silhouette = fit_kmeans(X, k=3)
    assert len(set(labels)) == 3
    assert -1 <= silhouette <= 1


def test_pca_projection_shape():
    patients = _toy_patients()
    ae = _toy_ae(patients)
    df = build_feature_table(patients, ae)
    X, _ = scale_features(df)
    coords, variance = pca_projection(X)
    assert coords.shape == (len(df), 2)
    assert len(variance) == 2


def test_cluster_labels_are_unique_when_clusters_are_distinct():
    patients = _toy_patients()
    ae = _toy_ae(patients)
    df = build_feature_table(patients, ae)
    X, _ = scale_features(df)
    _, labels, _ = fit_kmeans(X, k=3)
    profile = profile_clusters(df, labels)
    profile = label_all_clusters(profile, df)
    assert profile["business_label"].notna().all()
