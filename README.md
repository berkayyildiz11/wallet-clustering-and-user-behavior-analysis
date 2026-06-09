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

## Important project rule

The dataset contains a `FLAG` column. Do not use `FLAG` as an input feature for clustering. It should only be used after clustering as a validation or sanity-check signal.
