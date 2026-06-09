from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

FEATURES_PATH = Path("data/preprocessed/features_scaled.csv")
CLUSTERS_DIR = Path("outputs/clusters")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")

LABELS_FILE = CLUSTERS_DIR / "kmeans_labels.csv"
METRICS_FILE = TABLES_DIR / "kmeans_metrics.csv"

K_RANGE = range(2, 11)
RANDOM_SEED = 42


def load_scaled_features(features_path: str | Path = FEATURES_PATH) -> pd.DataFrame:
    """Load row-aligned scaled features for clustering."""
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Preprocessed features not found: {features_path}. "
            "Run `python -m src.data.run_preprocessing` first."
        )
    return pd.read_csv(features_path)


def evaluate_kmeans_range(
    features: pd.DataFrame,
    k_range: range = K_RANGE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Fit K-Means for each k and collect clustering metrics."""
    matrix = features.to_numpy()
    rows: list[dict[str, float | int]] = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = model.fit_predict(matrix)

        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(matrix, labels),
                "davies_bouldin_index": davies_bouldin_score(matrix, labels),
            }
        )

    return pd.DataFrame(rows)


def select_best_k(metrics: pd.DataFrame) -> int:
    """Pick k with the highest silhouette score; tie-break with lower Davies-Bouldin."""
    ranked = metrics.sort_values(
        by=["silhouette_score", "davies_bouldin_index"],
        ascending=[False, True],
    )
    return int(ranked.iloc[0]["k"])


def fit_kmeans(
    features: pd.DataFrame,
    n_clusters: int,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Fit final K-Means model and return cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    return model.fit_predict(features.to_numpy())


def build_labels_dataframe(labels: np.ndarray) -> pd.DataFrame:
    """Build row-aligned cluster label output with wallet_index."""
    if labels.shape[0] != 9_841:
        raise ValueError(f"Expected 9,841 labels, got {labels.shape[0]}.")

    if np.isnan(labels).any():
        raise ValueError("Cluster labels must not contain missing values.")

    return pd.DataFrame(
        {
            "wallet_index": np.arange(len(labels), dtype=int),
            "kmeans_cluster": labels.astype(int),
        }
    )


def save_metric_plots(metrics: pd.DataFrame, figures_dir: str | Path = FIGURES_DIR) -> None:
    """Save elbow and clustering metric comparison plots."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(metrics["k"], metrics["inertia"], marker="o")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_xticks(list(metrics["k"]))

    axes[1].plot(metrics["k"], metrics["silhouette_score"], marker="o", color="tab:green")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(list(metrics["k"]))

    axes[2].plot(metrics["k"], metrics["davies_bouldin_index"], marker="o", color="tab:orange")
    axes[2].set_title("Davies-Bouldin Index")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Index (lower is better)")
    axes[2].set_xticks(list(metrics["k"]))

    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["inertia"], marker="o", label="Inertia")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Plot")
    ax.set_xticks(list(metrics["k"]))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_kmeans_clustering(
    features_path: str | Path = FEATURES_PATH,
    clusters_dir: str | Path = CLUSTERS_DIR,
    tables_dir: str | Path = TABLES_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    k_range: range = K_RANGE,
    random_seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Run K-Means evaluation, select best k, and save labels, metrics, and plots."""
    features = load_scaled_features(features_path)
    metrics = evaluate_kmeans_range(features, k_range=k_range, random_seed=random_seed)
    best_k = select_best_k(metrics)
    labels = fit_kmeans(features, n_clusters=best_k, random_seed=random_seed)
    labels_df = build_labels_dataframe(labels)

    clusters_dir = Path(clusters_dir)
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels_df.to_csv(clusters_dir / "kmeans_labels.csv", index=False)
    metrics.to_csv(tables_dir / "kmeans_metrics.csv", index=False)
    save_metric_plots(metrics, figures_dir=figures_dir)

    cluster_counts = labels_df["kmeans_cluster"].value_counts().sort_index()
    print("K-Means clustering complete.")
    print(f"Rows evaluated: {len(features):,}")
    print(f"Best k (highest silhouette): {best_k}")
    print(f"Labels written to: {clusters_dir / 'kmeans_labels.csv'}")
    print(f"Metrics written to: {tables_dir / 'kmeans_metrics.csv'}")
    print(f"Figures written to: {figures_dir}/")
    print("Cluster sizes:")
    print(cluster_counts.to_string())

    return {
        "best_k": best_k,
        "metrics": metrics,
        "labels": labels_df,
        "cluster_counts": cluster_counts,
    }


if __name__ == "__main__":
    run_kmeans_clustering()
