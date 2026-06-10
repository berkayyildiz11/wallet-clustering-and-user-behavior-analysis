import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt


METRICS_PATH = "outputs/tables/dbscan_metrics.csv"
OUTPUT_PATH = "outputs/tables/best_dbscan_selection.csv"


def main():
    df = pd.read_csv(METRICS_PATH)

    # En az 3 cluster şartı
    candidates = df[df["n_clusters"] >= 3].copy()

    # Geçersiz metrikleri çıkar
    candidates = candidates.dropna(
        subset=["silhouette_score", "davies_bouldin_index"]
    )

    # Normalize edilmiş skorlar
    scaler = MinMaxScaler()

    candidates["silhouette_norm"] = scaler.fit_transform(
        candidates[["silhouette_score"]]
    )

    # DBI düşük olması iyi olduğu için ters çeviriyoruz
    candidates["dbi_norm"] = 1 - scaler.fit_transform(
        candidates[["davies_bouldin_index"]]
    )

    candidates["noise_norm"] = 1 - scaler.fit_transform(
        candidates[["n_noise"]]
    )

    # Cluster sayısı fazla olsun ama sadece buna göre seçmesin
    candidates["cluster_norm"] = scaler.fit_transform(
        candidates[["n_clusters"]]
    )

    # Toplam skor
    candidates["final_score"] = (
        0.45 * candidates["silhouette_norm"]
        + 0.30 * candidates["dbi_norm"]
        + 0.15 * candidates["noise_norm"]
        + 0.10 * candidates["cluster_norm"]
    )

    ranked = candidates.sort_values(
        "final_score",
        ascending=False
    )

    ranked.to_csv(OUTPUT_PATH, index=False)

    os.makedirs("outputs/figures", exist_ok=True)

    top10 = ranked.head(10)

    labels = (
        "eps="
        + top10["eps"].astype(str)
        + "\nmin="
        + top10["min_samples"].astype(int).astype(str)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(labels, top10["final_score"])

    for i, score in enumerate(top10["final_score"]):
        plt.text(score, i, f"{score:.3f}", va="center")

    plt.title("Top 10 DBSCAN Configurations")
    plt.xlabel("Final Score")
    plt.ylabel("(eps, min_samples)")

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/dbscan_top10_configs.png",
        dpi=300
    )

    plt.close()

    print("\nTop 10 DBSCAN configurations:")
    print(
        ranked[
            [
                "eps",
                "min_samples",
                "n_clusters",
                "n_noise",
                "silhouette_score",
                "davies_bouldin_index",
                "final_score"
            ]
        ].head(10)
    )

    best = ranked.iloc[0]

    print("\nBest DBSCAN configuration:")
    print(f"eps = {best['eps']}")
    print(f"min_samples = {int(best['min_samples'])}")
    print(f"n_clusters = {int(best['n_clusters'])}")
    print(f"n_noise = {int(best['n_noise'])}")
    print(f"silhouette_score = {best['silhouette_score']}")
    print(f"davies_bouldin_index = {best['davies_bouldin_index']}")
    print(f"final_score = {best['final_score']}")


if __name__ == "__main__":
    main()