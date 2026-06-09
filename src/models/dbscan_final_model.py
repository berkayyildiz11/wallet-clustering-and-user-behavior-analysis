import os
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


FEATURES_PATH = "data/preprocessed/features_scaled.csv"
OUTPUT_CLUSTER_PATH = "outputs/clusters/dbscan_labels.csv"


def main():
    os.makedirs("outputs/clusters", exist_ok=True)

    X = pd.read_csv(FEATURES_PATH)

    print("Features shape:", X.shape)

    final_model = DBSCAN(
        eps=5.0,
        min_samples=25
    )

    final_labels = final_model.fit_predict(X)

    print("\nFinal selected parameters:")
    print("eps = 5.0")

    print("min_samples = 25")

    print("\nCluster distribution:")
    print(pd.Series(final_labels).value_counts().sort_index())


    cluster_counts = pd.Series(final_labels).value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    cluster_counts.plot(kind="bar")

    plt.title("DBSCAN Cluster Size Distribution")
    plt.xlabel("DBSCAN Cluster")
    plt.ylabel("Wallet Count")
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/dbscan_cluster_distribution.png", dpi=300)
    plt.close()

    labels_df = pd.DataFrame({
        "wallet_index": range(len(final_labels)),
        "dbscan_cluster": final_labels,
        "is_noise": final_labels == -1
    })

    labels_df.to_csv(OUTPUT_CLUSTER_PATH, index=False)

    print("\nSaved:")
    print(OUTPUT_CLUSTER_PATH)



if __name__ == "__main__":
    main()