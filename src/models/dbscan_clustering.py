import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


FEATURES_PATH = "data/preprocessed/features_scaled.csv"
OUTPUT_METRICS_PATH = "outputs/tables/dbscan_metrics.csv"


def main():
    os.makedirs("outputs/tables", exist_ok=True)

    X = pd.read_csv(FEATURES_PATH)

    print("Features shape:", X.shape)

    results = []

    eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    min_samples_values = [5, 10, 20, 25, 30, 40, 50]

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            if n_clusters > 1:
                silhouette = silhouette_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)
            else:
                silhouette = None
                davies_bouldin = None

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette_score": silhouette,
                "davies_bouldin_index": davies_bouldin
            })

            print(
                f"eps={eps}, min_samples={min_samples}, "
                f"clusters={n_clusters}, noise={n_noise}, "
                f"silhouette={silhouette}, DBI={davies_bouldin}"
            )

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False)

    print("\nSaved:")
    print(OUTPUT_METRICS_PATH)


if __name__ == "__main__":
    main()