"""Run K-Means patient segmentation end-to-end: load data, standardize
features, pick k via the elbow method + silhouette score, fit K-Means,
profile and business-label each cluster, project to 2D with PCA, and
persist cluster assignments back to SQLite for the dashboard/chat agent.

Run: python -m src.clustering.run_clustering
"""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import DEFAULT_DB_PATH
from src.clustering.patient_clustering import (
    build_feature_table, scale_features, elbow_method, fit_kmeans,
    pca_projection, profile_clusters, label_all_clusters, FEATURE_COLUMNS,
)

PLOTS_DIR = PROJECT_ROOT / "models" / "cluster_plots"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "clustering_results.json"


def plot_elbow(elbow: dict, save_path: Path):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(elbow["k_values"], elbow["inertias"], marker="o", color="#4C72B0", label="Inertia")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(elbow["k_values"], elbow["silhouette_scores"], marker="s", color="#DD8452", label="Silhouette score")
    ax2.set_ylabel("Silhouette score", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    fig.suptitle("K-Means elbow method: inertia and silhouette score vs. k")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(coords, labels, cluster_names: dict, explained_variance, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        ax.scatter(coords[mask, 0], coords[mask, 1], s=25, alpha=0.7, color=palette[cluster_id % len(palette)], label=name)

    ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% variance)")
    ax.set_title("Patient clusters (PCA 2D projection)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run(db_path: Path = DEFAULT_DB_PATH, k_range=range(2, 9), final_k_candidates=(3, 4)):
    conn = sqlite3.connect(db_path)
    patients = pd.read_sql("SELECT * FROM patients", conn)
    ae = pd.read_sql("SELECT * FROM adverse_events", conn)

    df = build_feature_table(patients, ae)
    X, scaler = scale_features(df)

    print("Running elbow method (k=2..8)...")
    elbow = elbow_method(X, k_range)
    plot_elbow(elbow, PLOTS_DIR / "elbow_method.png")

    # Pick k with the best silhouette score among the requested candidate range (3-4)
    candidate_scores = {
        k: elbow["silhouette_scores"][elbow["k_values"].index(k)] for k in final_k_candidates
    }
    best_k = max(candidate_scores, key=candidate_scores.get)
    print(f"Silhouette scores for candidates {final_k_candidates}: {candidate_scores} -> choosing k={best_k}")

    km, labels, silhouette = fit_kmeans(X, best_k)
    print(f"Final K-Means: k={best_k}, inertia={km.inertia_:.1f}, silhouette={silhouette:.4f}")

    profile = profile_clusters(df, labels)
    profile = label_all_clusters(profile, df)
    print("\nCluster profiles:")
    print(profile.to_string(index=False))

    cluster_names = dict(zip(profile["cluster"], profile["business_label"]))

    coords, explained_variance = pca_projection(X)
    pca_plot_path = PLOTS_DIR / "pca_clusters.png"
    plot_pca_scatter(coords, labels, cluster_names, explained_variance, pca_plot_path)

    # Persist patient -> cluster assignment back to SQLite
    assignments = df[["patient_id"]].copy()
    assignments["cluster"] = labels
    assignments["cluster_label"] = assignments["cluster"].map(cluster_names)
    assignments.to_sql("patient_clusters", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"\nCluster assignments written to 'patient_clusters' table in {db_path}")

    results = {
        "chosen_k": best_k,
        "silhouette_score": round(float(silhouette), 4),
        "inertia": round(float(km.inertia_), 2),
        "elbow_curve": elbow,
        "feature_columns": FEATURE_COLUMNS,
        "cluster_profiles": profile.to_dict(orient="records"),
        "pca_explained_variance": [round(float(v), 4) for v in explained_variance],
        "elbow_plot_path": str((PLOTS_DIR / "elbow_method.png").relative_to(PROJECT_ROOT)),
        "pca_plot_path": str(pca_plot_path.relative_to(PROJECT_ROOT)),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run()
