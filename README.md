# wallet-clustering-and-user-behavior-analysis
**Wallet Clustering and User Behavior Analysis on Ethereum** explores blockchain transaction data to identify behavioral patterns among wallets. Using feature engineering and unsupervised machine learning (K-Means, DBSCAN), the project clusters Ethereum addresses into user groups such as whales, DeFi participants, and automated actors.

## Dataset setup

This project uses the Kaggle Ethereum Fraud Detection Dataset:

https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

Raw dataset files are stored locally under:

```text
data/raw/
```

They are ignored by Git, so every teammate should download the dataset locally.

### 1. Kaggle credentials

Create Kaggle credentials using one of these options:

```bash
kaggle auth login
```

or generate an API token from Kaggle settings and export it:

```bash
export KAGGLE_API_TOKEN=your_token_here
```

### 2. Download with uv

Run this command from the project root:

```bash
uv run python -m src.data.download
```

After downloading, the expected dataset path is:

```text
data/raw/transaction_dataset.csv
```

### 3. Run preprocessing

Run this command from the project root:

```bash
uv run python -m src.data.run_preprocessing
```

This creates intermediate cleaned files under:

```text
data/interim/
```

and the final scaled clustering input under:

```text
data/preprocessed/
```

### 4. Create shared train/validation/test splits

Run this command from the project root:

```bash
uv run python -m src.data.create_splits
```

This creates deterministic row-level split assignments under:

```text
data/splits/
```

The split file is useful for XGBoost/SHAP evaluation and shared stability checks. K-Means and DBSCAN can still be fit on the full preprocessed dataset because they are unsupervised clustering methods.

## Important project rule

The dataset contains a `FLAG` column. Do not use `FLAG` as an input feature for clustering. It should only be used after clustering as a validation or sanity-check signal.

## DBSCAN Clustering and Validation Workflow

This section describes the DBSCAN workflow.

### Required input files

Before running DBSCAN, preprocessing must be completed. The following files are required:

```text
data/preprocessed/features_scaled.csv
data/interim/wallet_metadata.csv
```

`features_scaled.csv` is used as the DBSCAN input.
`wallet_metadata.csv` contains `address` and `flag`. The `flag` column must not be used during clustering. It is only used after clustering for validation.

---

### 1. Run DBSCAN grid search

This step tests different `eps` and `min_samples` values and records clustering metrics.

```bash
uv run python -m src.models.dbscan_clustering
```

Output:

```text
outputs/tables/dbscan_metrics.csv
```

This file contains:

```text
eps
min_samples
n_clusters
n_noise
silhouette_score
davies_bouldin_index
```

---

### 2. Select the best DBSCAN configuration

This step reads `dbscan_metrics.csv`, applies the minimum 3-cluster condition, normalizes the metrics, and ranks DBSCAN configurations using a weighted final score.

```bash
uv run python -m src.models.select_best_dbscan
```

Output:

```text
outputs/tables/best_dbscan_selection.csv
outputs/figures/dbscan_top10_configs.png
```

Selection criteria:

```text
Minimum number of clusters: 3
Higher Silhouette Score is better
Lower Davies-Bouldin Index is better
Lower noise count is better
Cluster diversity is considered
```

The final selected DBSCAN configuration is:

```text
eps = 5.0
min_samples = 50
```

---

### 3. Run final DBSCAN model

This step runs DBSCAN using the selected final parameters and saves the cluster labels.

```bash
uv run python -m src.models.dbscan_final_model
```

Output:

```text
outputs/clusters/dbscan_labels.csv
outputs/figures/dbscan_cluster_distribution.png
```

The cluster label file contains:

```text
wallet_index
dbscan_cluster
is_noise
```

DBSCAN noise points are represented with cluster label `-1`.

---

### 4. Run post-clustering flag validation

This step checks whether flagged wallets concentrate in specific DBSCAN clusters or in the DBSCAN noise group.

```bash
uv run python -m src.models.dbscan_flag_validation
```

Output:

```text
outputs/tables/dbscan_flag_validation.csv
outputs/tables/dbscan_noise_flag_check.csv
outputs/figures/dbscan_fraud_rate.png
```

Important rule:

```text
flag is not used as a model input.
flag is only used after clustering for sanity-check validation.
```

---

### DBSCAN output summary

The final DBSCAN model uses:

```text
eps = 5.0
min_samples = 50
```

It produces:

```text
3 clusters
1 noise group
```

The final cluster labels are saved in:

```text
outputs/clusters/dbscan_labels.csv
```

This file is row-aligned with the original feature files and can be used for later interpretation and comparison.

## Cluster Interpretation Workflow

This step trains a tree-based model to predict K-Means pseudo-labels, then uses SHAP to explain which features distinguish the clusters.

Run this command from the project root:

```bash
uv run python -m src.interpretation.cluster_interpretation
```

Required inputs:

```text
data/interim/features_clean.csv
data/splits/split_indices.csv
outputs/clusters/kmeans_labels.csv
```

Outputs:

```text
outputs/tables/kmeans_pseudolabel_model_metrics.csv
outputs/tables/kmeans_pseudolabel_classification_report.csv
outputs/tables/kmeans_pseudolabel_model_predictions.csv
outputs/tables/kmeans_cluster_profiles.csv
outputs/tables/kmeans_cluster_behavior_interpretations.csv
outputs/tables/kmeans_cluster_top_features.csv
outputs/tables/kmeans_shap_global_importance.csv
outputs/figures/kmeans_pseudolabel_model_global_shap_importance.png
outputs/figures/kmeans_cluster_0_shap_importance.png
outputs/figures/kmeans_cluster_1_shap_importance.png
outputs/figures/kmeans_cluster_2_shap_importance.png
outputs/figures/kmeans_interpretation_cluster_sizes.png
```

XGBoost is the preferred pseudo-label model. On macOS, XGBoost may require the OpenMP runtime:

```bash
brew install libomp
```

If XGBoost cannot load, the script falls back to a SHAP-compatible random forest model so interpretation work can continue.
