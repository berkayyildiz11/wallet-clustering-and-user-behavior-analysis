from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)

FEATURES_PATH = Path("data/preprocessed/features_scaled.csv")
METADATA_PATH = Path("data/interim/wallet_metadata.csv")
CLUSTERS_DIR = Path("outputs/clusters")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")

LABELS_FILE = CLUSTERS_DIR / "kmeans_labels.csv"
METRICS_FILE = TABLES_DIR / "kmeans_metrics.csv"
DISTRIBUTIONS_FILE = TABLES_DIR / "kmeans_cluster_distributions.csv"
CLUSTER_DETAILS_FILE = TABLES_DIR / "kmeans_cluster_details.csv"
FLAG_VALIDATION_FILE = TABLES_DIR / "kmeans_cluster_flag_validation.csv"
ALL_LABELS_FILE = CLUSTERS_DIR / "kmeans_labels_all_k.csv"

K_RANGE = range(2, 11)
RANDOM_SEED = 42


@dataclass
class KMeansRunResult:
    k: int
    model: KMeans
    labels: np.ndarray
    metrics: dict[str, float | int]
    distributions: pd.DataFrame
    cluster_details: pd.DataFrame


def load_scaled_features(features_path: str | Path = FEATURES_PATH) -> pd.DataFrame:
    """Load row-aligned scaled features for clustering."""
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Preprocessed features not found: {features_path}. "
            "Run `python -m src.data.run_preprocessing` first."
        )
    return pd.read_csv(features_path)


def load_wallet_metadata(metadata_path: str | Path = METADATA_PATH) -> pd.DataFrame:
    """Load row-aligned wallet metadata for post-clustering flag validation only."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Wallet metadata not found: {metadata_path}. "
            "Run `python -m src.data.run_preprocessing` first."
        )

    metadata = pd.read_csv(metadata_path)
    if "flag" not in metadata.columns:
        raise ValueError("wallet_metadata.csv must contain a `flag` column.")

    return metadata


def build_cluster_flag_validation(
    runs: dict[int, KMeansRunResult],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare cluster assignments with ground-truth fraud flags from metadata."""
    flags = metadata["flag"].to_numpy()
    if np.isnan(flags).any():
        raise ValueError("Flag column contains missing values.")

    total_wallets = len(flags)
    total_fraud = int(flags.sum())
    total_non_fraud = int(total_wallets - total_fraud)
    global_fraud_rate = round(total_fraud / total_wallets * 100, 4)

    rows: list[dict[str, float | int]] = []
    for k, run in sorted(runs.items()):
        labels = run.labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_flags = flags[mask]
            fraud_count = int(cluster_flags.sum())
            non_fraud_count = int((cluster_flags == 0).sum())
            cluster_size = int(mask.sum())

            rows.append(
                {
                    "k": k,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "non_fraud_count": non_fraud_count,
                    "fraud_count": fraud_count,
                    "fraud_pct_in_cluster": round(fraud_count / cluster_size * 100, 4)
                    if cluster_size
                    else 0.0,
                    "fraud_share_of_all_fraud_pct": round(fraud_count / total_fraud * 100, 4)
                    if total_fraud
                    else 0.0,
                    "cluster_share_of_dataset_pct": round(cluster_size / total_wallets * 100, 4),
                    "fraud_rate_vs_global_pct": round(
                        fraud_count / cluster_size * 100 - global_fraud_rate, 4
                    )
                    if cluster_size
                    else 0.0,
                }
            )

    baseline = {
        "total_wallets": total_wallets,
        "total_fraud": total_fraud,
        "total_non_fraud": total_non_fraud,
        "global_fraud_rate_pct": global_fraud_rate,
    }
    return pd.DataFrame(rows), baseline


def _cluster_size_summary(labels: np.ndarray) -> dict[str, float | int]:
    counts = np.bincount(labels)
    return {
        "n_clusters": int(len(counts)),
        "smallest_cluster_size": int(counts.min()),
        "largest_cluster_size": int(counts.max()),
        "cluster_size_mean": float(counts.mean()),
        "cluster_size_std": float(counts.std(ddof=0)),
        "cluster_size_cv": float(counts.std(ddof=0) / counts.mean()) if counts.mean() else 0.0,
    }


def _build_cluster_distributions(k: int, labels: np.ndarray) -> pd.DataFrame:
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    return pd.DataFrame(
        {
            "k": k,
            "cluster_id": counts.index.astype(int),
            "count": counts.values.astype(int),
            "pct": (counts.values / total * 100).round(4),
        }
    )


def _build_cluster_details(
    k: int,
    matrix: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
) -> pd.DataFrame:
    sample_silhouettes = silhouette_samples(matrix, labels)
    rows: list[dict[str, float | int]] = []

    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_points = matrix[mask]
        centroid = model.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)

        rows.append(
            {
                "k": k,
                "cluster_id": cluster_id,
                "count": int(mask.sum()),
                "pct": round(mask.sum() / len(labels) * 100, 4),
                "inertia": float(np.sum(distances**2)),
                "mean_distance_to_centroid": float(distances.mean()),
                "max_distance_to_centroid": float(distances.max()),
                "mean_silhouette": float(sample_silhouettes[mask].mean()),
            }
        )

    return pd.DataFrame(rows)


def evaluate_kmeans_at_k(
    matrix: np.ndarray,
    k: int,
    random_seed: int = RANDOM_SEED,
) -> KMeansRunResult:
    """Fit K-Means for a single k and collect full evaluation details."""
    model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = model.fit_predict(matrix)

    metrics = {
        "k": k,
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "silhouette_score": float(silhouette_score(matrix, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(matrix, labels)),
        **_cluster_size_summary(labels),
    }

    return KMeansRunResult(
        k=k,
        model=model,
        labels=labels,
        metrics=metrics,
        distributions=_build_cluster_distributions(k, labels),
        cluster_details=_build_cluster_details(k, matrix, labels, model),
    )


def evaluate_kmeans_range(
    features: pd.DataFrame,
    k_range: range = K_RANGE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, KMeansRunResult]]:
    """Fit K-Means for each k and collect metrics, distributions, and per-cluster details."""
    matrix = features.to_numpy()
    runs: dict[int, KMeansRunResult] = {}

    for k in k_range:
        runs[k] = evaluate_kmeans_at_k(matrix, k, random_seed=random_seed)

    metrics = pd.DataFrame([run.metrics for run in runs.values()])
    distributions = pd.concat([run.distributions for run in runs.values()], ignore_index=True)
    cluster_details = pd.concat([run.cluster_details for run in runs.values()], ignore_index=True)

    return metrics, distributions, cluster_details, runs


def select_best_k(metrics: pd.DataFrame) -> int:
    """Pick k with the highest silhouette score; tie-break with lower Davies-Bouldin."""
    ranked = metrics.sort_values(
        by=["silhouette_score", "davies_bouldin_index"],
        ascending=[False, True],
    )
    return int(ranked.iloc[0]["k"])


def build_labels_dataframe(labels: np.ndarray, expected_rows: int | None = None) -> pd.DataFrame:
    """Build row-aligned cluster label output with wallet_index."""
    if expected_rows is not None and labels.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} labels, got {labels.shape[0]:,}.")

    if np.isnan(labels).any():
        raise ValueError("Cluster labels must not contain missing values.")

    return pd.DataFrame(
        {
            "wallet_index": np.arange(len(labels), dtype=int),
            "kmeans_cluster": labels.astype(int),
        }
    )


def build_all_k_labels_dataframe(runs: dict[int, KMeansRunResult]) -> pd.DataFrame:
    """Build long-format labels for every evaluated k value."""
    frames: list[pd.DataFrame] = []
    n_rows = next(iter(runs.values())).labels.shape[0]

    for k, run in sorted(runs.items()):
        frames.append(
            pd.DataFrame(
                {
                    "wallet_index": np.arange(n_rows, dtype=int),
                    "k": k,
                    "kmeans_cluster": run.labels.astype(int),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def save_metric_plots(metrics: pd.DataFrame, figures_dir: str | Path = FIGURES_DIR) -> None:
    """Save elbow and clustering metric comparison plots."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(metrics["k"], metrics["inertia"], marker="o")
    axes[0, 0].set_title("Elbow Method (Inertia)")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].set_xticks(list(metrics["k"]))
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(metrics["k"], metrics["silhouette_score"], marker="o", color="tab:green")
    axes[0, 1].set_title("Silhouette Score")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_xticks(list(metrics["k"]))
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(metrics["k"], metrics["davies_bouldin_index"], marker="o", color="tab:orange")
    axes[1, 0].set_title("Davies-Bouldin Index")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Index (lower is better)")
    axes[1, 0].set_xticks(list(metrics["k"]))
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(metrics["k"], metrics["calinski_harabasz_score"], marker="o", color="tab:purple")
    axes[1, 1].set_title("Calinski-Harabasz Score")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Score (higher is better)")
    axes[1, 1].set_xticks(list(metrics["k"]))
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Plot")
    ax.set_xticks(list(metrics["k"]))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_flag_validation_plots(
    flag_validation: pd.DataFrame,
    baseline: dict[str, float | int],
    figures_dir: str | Path = FIGURES_DIR,
) -> None:
    """Save fraud concentration plots for each evaluated k."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    global_rate = float(baseline["global_fraud_rate_pct"])

    k_values = sorted(flag_validation["k"].unique())
    n_k = len(k_values)
    n_cols = 3
    n_rows = int(np.ceil(n_k / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for index, k in enumerate(k_values):
        subset = flag_validation[flag_validation["k"] == k]
        ax = axes_flat[index]
        x = subset["cluster_id"].astype(str)
        ax.bar(x, subset["non_fraud_count"], label="Non-fraud (flag=0)", color="tab:blue")
        ax.bar(
            x,
            subset["fraud_count"],
            bottom=subset["non_fraud_count"],
            label="Fraud (flag=1)",
            color="tab:red",
        )
        ax.set_title(f"k={k} fraud vs non-fraud by cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Wallet count")
        if index == 0:
            ax.legend(fontsize=8)

    for index in range(len(k_values), len(axes_flat)):
        axes_flat[index].axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_cluster_flag_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for index, k in enumerate(k_values):
        subset = flag_validation[flag_validation["k"] == k]
        ax = axes_flat[index]
        bars = ax.bar(
            subset["cluster_id"].astype(str),
            subset["fraud_pct_in_cluster"],
            color="tab:red",
            alpha=0.8,
        )
        ax.axhline(global_rate, color="black", linestyle="--", label=f"Global rate ({global_rate:.1f}%)")
        ax.set_title(f"k={k} fraud rate by cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Fraud rate (%)")
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in subset["fraud_pct_in_cluster"]], fontsize=8)
        if index == 0:
            ax.legend(fontsize=8)

    for index in range(len(k_values), len(axes_flat)):
        axes_flat[index].axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_cluster_fraud_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_distribution_plots(
    distributions: pd.DataFrame,
    figures_dir: str | Path = FIGURES_DIR,
) -> None:
    """Save per-k cluster size distribution plots and a combined heatmap."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    k_values = sorted(distributions["k"].unique())
    n_k = len(k_values)
    n_cols = 3
    n_rows = int(np.ceil(n_k / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for index, k in enumerate(k_values):
        subset = distributions[distributions["k"] == k]
        ax = axes_flat[index]
        bars = ax.bar(subset["cluster_id"].astype(str), subset["pct"], color="steelblue")
        ax.set_title(f"k={k} cluster distribution")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Share (%)")
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in subset["pct"]], fontsize=8)

    for index in range(len(k_values), len(axes_flat)):
        axes_flat[index].axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_cluster_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pivot = distributions.pivot(index="cluster_id", columns="k", values="pct").fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(4, pivot.shape[0] * 0.4)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("k")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Cluster share (%) by k")
    fig.colorbar(im, ax=ax, label="pct")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_cluster_distribution_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_flag_validation_summary(
    flag_validation: pd.DataFrame,
    baseline: dict[str, float | int],
    best_k: int,
) -> None:
    """Print fraud flag sanity-check results against ground-truth metadata."""
    print("=== Ground-truth fraud baseline (wallet_metadata) ===")
    print(f"Total wallets: {baseline['total_wallets']:,}")
    print(f"Non-fraud (flag=0): {baseline['total_non_fraud']:,}")
    print(f"Fraud (flag=1): {baseline['total_fraud']:,}")
    print(f"Global fraud rate: {baseline['global_fraud_rate_pct']:.4f}%")
    print()
    print("=== Fraud counts by cluster and k ===")
    for k in sorted(flag_validation["k"].unique()):
        subset = flag_validation[flag_validation["k"] == k]
        print(f"\nk={k}")
        print(
            subset[
                [
                    "cluster_id",
                    "cluster_size",
                    "non_fraud_count",
                    "fraud_count",
                    "fraud_pct_in_cluster",
                    "fraud_share_of_all_fraud_pct",
                    "fraud_rate_vs_global_pct",
                ]
            ].to_string(index=False)
        )
    print()
    print(f"=== Fraud counts for best k={best_k} ===")
    best_subset = flag_validation[flag_validation["k"] == best_k]
    print(
        best_subset[
            [
                "cluster_id",
                "cluster_size",
                "non_fraud_count",
                "fraud_count",
                "fraud_pct_in_cluster",
                "fraud_share_of_all_fraud_pct",
            ]
        ].to_string(index=False)
    )


def print_detailed_summary(
    metrics: pd.DataFrame,
    distributions: pd.DataFrame,
    cluster_details: pd.DataFrame,
    best_k: int,
) -> None:
    """Print a readable summary of all evaluated k values."""
    print("K-Means clustering complete.")
    print()
    print("=== Global metrics by k ===")
    print(
        metrics[
            [
                "k",
                "inertia",
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_score",
                "n_iter",
                "smallest_cluster_size",
                "largest_cluster_size",
                "cluster_size_cv",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Best k (highest silhouette): {best_k}")
    print()
    print("=== Cluster distributions by k ===")
    for k in sorted(distributions["k"].unique()):
        subset = distributions[distributions["k"] == k]
        print(f"\nk={k}")
        print(subset[["cluster_id", "count", "pct"]].to_string(index=False))
    print()
    print(f"=== Per-cluster details for best k={best_k} ===")
    best_details = cluster_details[cluster_details["k"] == best_k]
    print(
        best_details[
            [
                "cluster_id",
                "count",
                "pct",
                "inertia",
                "mean_distance_to_centroid",
                "mean_silhouette",
            ]
        ].to_string(index=False)
    )


def run_kmeans_clustering(
    features_path: str | Path = FEATURES_PATH,
    metadata_path: str | Path = METADATA_PATH,
    clusters_dir: str | Path = CLUSTERS_DIR,
    tables_dir: str | Path = TABLES_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    k_range: range = K_RANGE,
    random_seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Run K-Means evaluation, select best k, and save labels, metrics, and plots."""
    features = load_scaled_features(features_path)
    metadata = load_wallet_metadata(metadata_path)
    if len(metadata) != len(features):
        raise ValueError(
            f"Metadata rows ({len(metadata):,}) do not match feature rows ({len(features):,})."
        )

    metrics, distributions, cluster_details, runs = evaluate_kmeans_range(
        features,
        k_range=k_range,
        random_seed=random_seed,
    )
    flag_validation, fraud_baseline = build_cluster_flag_validation(runs, metadata)
    best_k = select_best_k(metrics)
    best_labels = runs[best_k].labels
    labels_df = build_labels_dataframe(best_labels, expected_rows=len(features))
    all_k_labels_df = build_all_k_labels_dataframe(runs)

    clusters_dir = Path(clusters_dir)
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels_df.to_csv(clusters_dir / "kmeans_labels.csv", index=False)
    all_k_labels_df.to_csv(clusters_dir / "kmeans_labels_all_k.csv", index=False)
    metrics.to_csv(tables_dir / "kmeans_metrics.csv", index=False)
    distributions.to_csv(tables_dir / "kmeans_cluster_distributions.csv", index=False)
    cluster_details.to_csv(tables_dir / "kmeans_cluster_details.csv", index=False)
    flag_validation.to_csv(tables_dir / "kmeans_cluster_flag_validation.csv", index=False)
    save_metric_plots(metrics, figures_dir=figures_dir)
    save_distribution_plots(distributions, figures_dir=figures_dir)
    save_flag_validation_plots(flag_validation, fraud_baseline, figures_dir=figures_dir)

    print_detailed_summary(metrics, distributions, cluster_details, best_k)
    print()
    print_flag_validation_summary(flag_validation, fraud_baseline, best_k)
    print()
    print(f"Best-k labels written to: {clusters_dir / 'kmeans_labels.csv'}")
    print(f"All-k labels written to: {clusters_dir / 'kmeans_labels_all_k.csv'}")
    print(f"Metrics written to: {tables_dir / 'kmeans_metrics.csv'}")
    print(f"Distributions written to: {tables_dir / 'kmeans_cluster_distributions.csv'}")
    print(f"Cluster details written to: {tables_dir / 'kmeans_cluster_details.csv'}")
    print(f"Flag validation written to: {tables_dir / 'kmeans_cluster_flag_validation.csv'}")
    print(f"Figures written to: {figures_dir}/")

    return {
        "best_k": best_k,
        "metrics": metrics,
        "distributions": distributions,
        "cluster_details": cluster_details,
        "flag_validation": flag_validation,
        "fraud_baseline": fraud_baseline,
        "labels": labels_df,
        "all_k_labels": all_k_labels_df,
        "runs": runs,
    }


if __name__ == "__main__":
    run_kmeans_clustering()
