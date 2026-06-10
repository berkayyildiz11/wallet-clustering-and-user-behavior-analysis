import os
import pandas as pd
import matplotlib.pyplot as plt


LABELS_PATH = "outputs/clusters/dbscan_labels.csv"
METADATA_PATH = "data/interim/wallet_metadata.csv"

OUTPUT_CLUSTER_VALIDATION = "outputs/tables/dbscan_flag_validation.csv"
OUTPUT_NOISE_VALIDATION = "outputs/tables/dbscan_noise_flag_check.csv"


def main():
    os.makedirs("outputs/tables", exist_ok=True)

    labels_df = pd.read_csv(LABELS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    print("Labels shape:", labels_df.shape)
    print("Metadata shape:", metadata.shape)

    if len(labels_df) != len(metadata):
        raise ValueError("labels_df ve metadata satır sayısı eşleşmiyor!")

    validation_df = labels_df.copy()

    if "flag" in metadata.columns:
        validation_df["flag"] = metadata["flag"]
    elif "FLAG" in metadata.columns:
        validation_df["flag"] = metadata["FLAG"]
    else:
        raise ValueError("metadata içinde flag veya FLAG sütunu bulunamadı!")

    cluster_validation = validation_df.groupby("dbscan_cluster")["flag"].agg(
        count="count",
        flagged_count="sum",
        flagged_rate="mean"
    ).reset_index()

    cluster_validation.to_csv(OUTPUT_CLUSTER_VALIDATION, index=False)

    plt.figure(figsize=(8, 5))

    plt.bar(
        cluster_validation["dbscan_cluster"].astype(str),
        cluster_validation["flagged_rate"] * 100
    )

    plt.title("Fraud Rate by DBSCAN Cluster")
    plt.xlabel("DBSCAN Cluster")
    plt.ylabel("Fraud Rate (%)")

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/dbscan_fraud_rate.png",
        dpi=300
    )

    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(
        cluster_validation["dbscan_cluster"].astype(str),
        cluster_validation["count"],
        alpha=0.7
    )

    ax1.set_xlabel("DBSCAN Cluster")
    ax1.set_ylabel("Wallet Count")

    ax2 = ax1.twinx()

    ax2.plot(
        cluster_validation["dbscan_cluster"].astype(str),
        cluster_validation["flagged_rate"] * 100,
        marker="o",
        linewidth=2
    )

    ax2.set_ylabel("Fraud Rate (%)")

    plt.title("DBSCAN Cluster Size and Fraud Rate")

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/dbscan_cluster_fraud_analysis.png",
        dpi=300
    )

    plt.close()


    noise_validation = validation_df.groupby("is_noise")["flag"].agg(
        count="count",
        flagged_count="sum",
        flagged_rate="mean"
    ).reset_index()

    noise_validation.to_csv(OUTPUT_NOISE_VALIDATION, index=False)

    print("\nCluster-based flag validation:")
    print(cluster_validation)

    print("\nNoise-based flag validation:")
    print(noise_validation)

    print("\nSaved:")
    print(OUTPUT_CLUSTER_VALIDATION)
    print(OUTPUT_NOISE_VALIDATION)


if __name__ == "__main__":
    main()