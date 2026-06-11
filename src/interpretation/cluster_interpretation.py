from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


FEATURES_CLEAN_PATH = Path("data/interim/features_clean.csv")
SPLITS_PATH = Path("data/splits/split_indices.csv")
KMEANS_LABELS_PATH = Path("outputs/clusters/kmeans_labels.csv")

TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")

MODEL_METRICS_FILE = TABLES_DIR / "kmeans_pseudolabel_model_metrics.csv"
CLASSIFICATION_REPORT_FILE = TABLES_DIR / "kmeans_pseudolabel_classification_report.csv"
CLUSTER_PROFILE_FILE = TABLES_DIR / "kmeans_cluster_profiles.csv"
BEHAVIOR_INTERPRETATION_FILE = TABLES_DIR / "kmeans_cluster_behavior_interpretations.csv"
TOP_FEATURES_FILE = TABLES_DIR / "kmeans_cluster_top_features.csv"
GLOBAL_SHAP_FILE = TABLES_DIR / "kmeans_shap_global_importance.csv"

RANDOM_SEED = 42
TOP_N_FEATURES = 8


def run_cluster_interpretation() -> None:
    features, labels, splits = load_inputs()
    dataset = build_modeling_dataset(features, labels, splits)

    x_train, y_train = split_xy(dataset, "train")
    x_validation, y_validation = split_xy(dataset, "validation")
    x_test, y_test = split_xy(dataset, "test")

    model = train_xgboost_classifier(x_train, y_train)

    metrics = evaluate_model(
        model=model,
        datasets={
            "train": (x_train, y_train),
            "validation": (x_validation, y_validation),
            "test": (x_test, y_test),
        },
    )

    shap_values = compute_shap_values(model, features)
    global_importance = build_global_shap_importance(features, shap_values)
    cluster_top_features = build_cluster_top_features(features, shap_values)
    cluster_profiles = build_cluster_profiles(features, labels, cluster_top_features)
    behavior_interpretations = build_behavior_interpretations(
        features,
        labels,
        cluster_top_features,
    )

    save_outputs(
        metrics,
        model,
        dataset,
        global_importance,
        cluster_top_features,
        cluster_profiles,
        behavior_interpretations,
    )
    save_figures(features, shap_values, global_importance, cluster_profiles)

    print("Cluster interpretation complete.")
    print(f"Metrics written to: {MODEL_METRICS_FILE}")
    print(f"Cluster profiles written to: {CLUSTER_PROFILE_FILE}")
    print(f"Behavior interpretations written to: {BEHAVIOR_INTERPRETATION_FILE}")
    print(f"Top SHAP features written to: {TOP_FEATURES_FILE}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (FEATURES_CLEAN_PATH, SPLITS_PATH, KMEANS_LABELS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    features = pd.read_csv(FEATURES_CLEAN_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    labels = pd.read_csv(KMEANS_LABELS_PATH)

    validate_inputs(features, labels, splits)
    return features, labels, splits


def validate_inputs(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:
    required_label_columns = {"wallet_index", "kmeans_cluster"}
    required_split_columns = {"wallet_index", "split"}

    if not required_label_columns.issubset(labels.columns):
        raise ValueError(f"{KMEANS_LABELS_PATH} must contain {required_label_columns}.")
    if not required_split_columns.issubset(splits.columns):
        raise ValueError(f"{SPLITS_PATH} must contain {required_split_columns}.")
    if len(features) != len(labels) or len(features) != len(splits):
        raise ValueError("Features, labels, and splits must have the same row count.")
    if labels["wallet_index"].duplicated().any() or splits["wallet_index"].duplicated().any():
        raise ValueError("wallet_index must be unique in label and split files.")
    if set(labels["wallet_index"]) != set(range(len(features))):
        raise ValueError("K-Means labels must cover every feature row exactly once.")
    if set(splits["wallet_index"]) != set(range(len(features))):
        raise ValueError("Splits must cover every feature row exactly once.")
    if features.isna().any().any():
        raise ValueError("features_clean.csv contains missing values.")


def build_modeling_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    features_with_index = features.copy()
    features_with_index.insert(0, "wallet_index", np.arange(len(features_with_index)))

    dataset = (
        features_with_index.merge(labels, on="wallet_index", how="inner")
        .merge(splits, on="wallet_index", how="inner")
        .sort_values("wallet_index")
        .reset_index(drop=True)
    )

    if len(dataset) != len(features):
        raise ValueError("Merged modeling dataset lost rows.")

    return dataset


def split_xy(dataset: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    split_dataset = dataset[dataset["split"] == split_name]
    x = split_dataset.drop(columns=["wallet_index", "kmeans_cluster", "split"])
    y = split_dataset["kmeans_cluster"]
    return x, y


def train_xgboost_classifier(x_train: pd.DataFrame, y_train: pd.Series) -> Any:
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=int(y_train.nunique()),
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    except Exception as error:
        print("XGBoost could not be imported or initialized.")
        print(f"Falling back to RandomForestClassifier for SHAP-compatible interpretation: {error}")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        )

    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: Any,
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    report_frames: list[pd.DataFrame] = []

    for split_name, (x, y) in datasets.items():
        predictions = model.predict(x)
        rows.append(
            {
                "split": split_name,
                "model_type": type(model).__name__,
                "rows": len(y),
                "accuracy": accuracy_score(y, predictions),
                "macro_f1": f1_score(y, predictions, average="macro"),
                "weighted_f1": f1_score(y, predictions, average="weighted"),
            }
        )

        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        report_df = (
            pd.DataFrame(report)
            .transpose()
            .reset_index()
            .rename(columns={"index": "class_or_average"})
        )
        report_df.insert(0, "split", split_name)
        report_frames.append(report_df)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(report_frames, ignore_index=True).to_csv(CLASSIFICATION_REPORT_FILE, index=False)
    return pd.DataFrame(rows)


def compute_shap_values(model: Any, features: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(features)

    if isinstance(values, list):
        return np.stack(values, axis=2)

    values_array = np.asarray(values)
    if values_array.ndim == 3:
        return values_array
    if values_array.ndim == 2:
        return values_array[:, :, np.newaxis]

    raise ValueError(f"Unexpected SHAP value shape: {values_array.shape}")


def build_global_shap_importance(
    features: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def build_cluster_top_features(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    top_n: int = TOP_N_FEATURES,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    n_classes = shap_values.shape[2]

    for cluster_id in range(n_classes):
        mean_abs_shap = np.abs(shap_values[:, :, cluster_id]).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "feature": features.columns[feature_index],
                    "mean_abs_shap": mean_abs_shap[feature_index],
                }
            )

    return pd.DataFrame(rows)


def build_cluster_profiles(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cluster_top_features: pd.DataFrame,
) -> pd.DataFrame:
    labeled_features = features.copy()
    labeled_features["kmeans_cluster"] = labels.sort_values("wallet_index")["kmeans_cluster"].to_numpy()

    rows: list[dict[str, float | int | str]] = []
    total_rows = len(labeled_features)

    for cluster_id in sorted(labeled_features["kmeans_cluster"].unique()):
        cluster_data = labeled_features[labeled_features["kmeans_cluster"] == cluster_id]
        top_features = cluster_top_features[
            cluster_top_features["cluster_id"] == cluster_id
        ]["feature"].head(5).tolist()

        profile: dict[str, float | int | str] = {
            "cluster_id": int(cluster_id),
            "size": int(len(cluster_data)),
            "dataset_share_pct": round(len(cluster_data) / total_rows * 100, 4),
            "top_shap_features": "; ".join(top_features),
            "interpretation_note": "Requires human interpretation from feature distributions and SHAP; do not treat as a ground-truth wallet class.",
        }

        for feature in top_features:
            profile[f"{feature}_cluster_median"] = cluster_data[feature].median()
            profile[f"{feature}_overall_median"] = labeled_features[feature].median()
            profile[f"{feature}_cluster_p75"] = cluster_data[feature].quantile(0.75)

        rows.append(profile)

    return pd.DataFrame(rows)


def build_behavior_interpretations(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cluster_top_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create cautious human-readable behavioral interpretations for each cluster."""
    labeled_features = features.copy()
    labeled_features["kmeans_cluster"] = labels.sort_values("wallet_index")["kmeans_cluster"].to_numpy()
    overall_medians = features.median()

    rows: list[dict[str, float | int | str]] = []
    for cluster_id in sorted(labeled_features["kmeans_cluster"].unique()):
        cluster_data = labeled_features[labeled_features["kmeans_cluster"] == cluster_id]
        medians = cluster_data.drop(columns=["kmeans_cluster"]).median()
        top_features = cluster_top_features[
            cluster_top_features["cluster_id"] == cluster_id
        ]["feature"].head(5).tolist()

        label, interpretation, caveat = infer_behavior_label(
            medians,
            overall_medians,
            cluster_size=len(cluster_data),
            total_size=len(labeled_features),
        )
        evidence = build_evidence_text(medians, overall_medians, top_features)

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(cluster_data)),
                "dataset_share_pct": round(len(cluster_data) / len(labeled_features) * 100, 4),
                "suggested_behavior_label": label,
                "cautious_interpretation": interpretation,
                "main_evidence": evidence,
                "top_shap_features": "; ".join(top_features),
                "caveat": caveat,
            }
        )

    return pd.DataFrame(rows)


def infer_behavior_label(
    medians: pd.Series,
    overall_medians: pd.Series,
    cluster_size: int,
    total_size: int,
) -> tuple[str, str, str]:
    high_balance = medians["total_ether_balance"] > overall_medians["total_ether_balance"] + 2
    receiver_heavy = (
        medians["received_tnx"] > overall_medians["received_tnx"] + 1
        and medians["sent_tnx"] <= overall_medians["sent_tnx"]
    )
    high_activity = (
        medians["sent_tnx"] > overall_medians["sent_tnx"] + 2
        and medians["total_transactions_including_tnx_to_create_contract"]
        > overall_medians["total_transactions_including_tnx_to_create_contract"] + 2
    )
    high_volume = (
        medians["total_ether_sent"] > overall_medians["total_ether_sent"] + 2
        and medians["total_ether_received"] > overall_medians["total_ether_received"] + 1
    )
    high_erc20 = medians["total_erc20_tnxs"] > overall_medians["total_erc20_tnxs"] + 2
    majority_cluster = cluster_size > 0.5 * total_size

    if high_activity and high_volume and high_erc20:
        return (
            "active-trader / DeFi-power-user-like wallets",
            "This cluster exhibits characteristics consistent with highly active wallets, such as active traders, DeFi power users, or automated high-activity accounts.",
            "This is not a verified DeFi/trader/bot label; it is inferred from high transaction activity, high Ether movement, ERC20 activity, and SHAP-supported feature separation.",
        )

    if high_balance and receiver_heavy:
        return (
            "receiver-heavy holder / whale-like wallets",
            "This cluster exhibits characteristics consistent with wallets that receive many transfers, send little, and maintain a relatively high positive Ether balance.",
            "This may include holders, accumulation wallets, deposit-style wallets, or whale-like wallets; the dataset does not provide ground-truth wallet identities.",
        )

    if majority_cluster:
        return (
            "retail / ordinary-transfer-like wallets",
            "This cluster exhibits characteristics consistent with the broad baseline population: moderate ETH transfer behavior, near-neutral balance, and limited ERC20 activity.",
            "Because this is the largest cluster, it should be interpreted as a baseline behavioral group rather than a single real-world user type.",
        )

    return (
        "mixed or anomalous behavior",
        "This cluster has a distinct feature profile but does not map cleanly to one behavioral category without additional wallet labels.",
        "Interpret this group cautiously and rely on feature distributions rather than a definitive class name.",
    )


def build_evidence_text(
    medians: pd.Series,
    overall_medians: pd.Series,
    top_features: list[str],
) -> str:
    evidence_parts: list[str] = []
    for feature in top_features:
        cluster_value = medians[feature]
        overall_value = overall_medians[feature]
        if cluster_value > overall_value:
            direction = "higher"
        elif cluster_value < overall_value:
            direction = "lower"
        else:
            direction = "similar"

        evidence_parts.append(
            f"{feature} is {direction} than the overall median "
            f"(cluster median={cluster_value:.4f}, overall median={overall_value:.4f})"
        )

    return "; ".join(evidence_parts)


def save_outputs(
    metrics: pd.DataFrame,
    model: Any,
    dataset: pd.DataFrame,
    global_importance: pd.DataFrame,
    cluster_top_features: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    behavior_interpretations: pd.DataFrame,
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(MODEL_METRICS_FILE, index=False)
    global_importance.to_csv(GLOBAL_SHAP_FILE, index=False)
    cluster_top_features.to_csv(TOP_FEATURES_FILE, index=False)
    cluster_profiles.to_csv(CLUSTER_PROFILE_FILE, index=False)
    behavior_interpretations.to_csv(BEHAVIOR_INTERPRETATION_FILE, index=False)

    predictions = dataset[["wallet_index", "split", "kmeans_cluster"]].copy()
    feature_columns = dataset.drop(columns=["wallet_index", "kmeans_cluster", "split"]).columns
    predictions["pseudolabel_model_predicted_cluster"] = model.predict(dataset[feature_columns])
    predictions.to_csv(TABLES_DIR / "kmeans_pseudolabel_model_predictions.csv", index=False)


def save_figures(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    global_importance: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    top_global = global_importance.head(15).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top_global["feature"], top_global["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Top Global SHAP Features for K-Means Pseudo-Labels")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_pseudolabel_model_global_shap_importance.png", dpi=300)
    plt.close()

    mean_abs_by_class = np.abs(shap_values).mean(axis=0)
    for cluster_id in range(mean_abs_by_class.shape[1]):
        top_indices = np.argsort(mean_abs_by_class[:, cluster_id])[::-1][:10]
        top = pd.DataFrame(
            {
                "feature": features.columns[top_indices],
                "mean_abs_shap": mean_abs_by_class[top_indices, cluster_id],
            }
        ).sort_values("mean_abs_shap", ascending=True)

        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Mean absolute SHAP value")
        plt.title(f"Top SHAP Features for K-Means Cluster {cluster_id}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"kmeans_cluster_{cluster_id}_shap_importance.png", dpi=300)
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(cluster_profiles["cluster_id"].astype(str), cluster_profiles["size"])
    plt.xlabel("K-Means cluster")
    plt.ylabel("Wallet count")
    plt.title("K-Means Cluster Sizes Used for Interpretation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_interpretation_cluster_sizes.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_cluster_interpretation()
