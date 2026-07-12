"""K-Means patient segmentation: standardized clinical features -> elbow
method for k selection -> K-Means -> cluster profiling -> PCA projection.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "age", "bmi", "baseline_systolic_bp", "baseline_diastolic_bp",
    "baseline_hba1c", "baseline_ldl", "baseline_egfr",
    "primary_outcome_score", "max_ae_severity_rank",
]

SEVERITY_RANK = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}


def build_feature_table(patients: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    """Join patient baseline/outcome data with each patient's worst
    adverse-event severity encountered during the trial."""
    ae_rank = ae.assign(severity_rank=ae["severity"].map(SEVERITY_RANK))
    max_severity = ae_rank.groupby("patient_id")["severity_rank"].max().rename("max_ae_severity_rank")
    df = patients.set_index("patient_id").join(max_severity).reset_index()
    df["max_ae_severity_rank"] = df["max_ae_severity_rank"].fillna(0)
    return df


def scale_features(df: pd.DataFrame, feature_columns=FEATURE_COLUMNS):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns])
    return X, scaler


def elbow_method(X: np.ndarray, k_range=range(2, 9)) -> dict:
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return {"k_values": list(k_range), "inertias": inertias, "silhouette_scores": silhouettes}


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    return km, labels, silhouette


def pca_projection(X: np.ndarray, seed: int = 42):
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    profiled = df.copy()
    profiled["cluster"] = labels

    agg = profiled.groupby("cluster").agg(
        n_patients=("patient_id", "count"),
        mean_age=("age", "mean"),
        mean_bmi=("bmi", "mean"),
        mean_baseline_hba1c=("baseline_hba1c", "mean"),
        mean_outcome_score=("primary_outcome_score", "mean"),
        mean_ae_severity_rank=("max_ae_severity_rank", "mean"),
        severe_ae_rate=("max_ae_severity_rank", lambda s: (s == 3).mean()),
        any_ae_rate=("max_ae_severity_rank", lambda s: (s > 0).mean()),
        dropout_rate=("dropout_flag", "mean"),
        pct_drug_x=("treatment_arm", lambda s: (s == "Drug X").mean()),
    ).round(3)
    return agg.reset_index()


def label_all_clusters(profile: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a unique business-friendly name per cluster, ranked relative to
    the other clusters in this run (not a fixed population threshold), so
    labels stay distinct even when several clusters look broadly similar.

    1. Any cluster with a materially elevated severe-AE rate is "AE-Prone"
       regardless of its outcome score — safety signal takes priority.
    2. Among the remaining clusters, the highest mean outcome score is
       "High Responders", the lowest is "Non-Responders", and anything in
       between is "Stable / Average Responders".
    """
    profile = profile.copy()
    severe_threshold = max(0.15, full_df.shape[0] and (full_df["max_ae_severity_rank"] == 3).mean() * 2)

    labels = pd.Series(index=profile.index, dtype=object)
    ae_prone_mask = profile["severe_ae_rate"] >= severe_threshold
    labels[ae_prone_mask] = "AE-Prone"

    remaining = profile.loc[~ae_prone_mask].sort_values("mean_outcome_score", ascending=False)
    if len(remaining) == 1:
        labels[remaining.index] = "Stable / Average Responders"
    elif len(remaining) > 1:
        ordered_idx = remaining.index.tolist()
        labels[ordered_idx[0]] = "High Responders"
        labels[ordered_idx[-1]] = "Non-Responders"
        for idx in ordered_idx[1:-1]:
            labels[idx] = "Stable / Average Responders"

    profile["business_label"] = labels
    return profile
