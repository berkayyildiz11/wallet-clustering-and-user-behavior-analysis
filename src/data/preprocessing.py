from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


LEAKAGE_AND_ID_COLUMNS = {
    "row_id",
    "dataset_index",
    "address",
    "flag",
}

CATEGORICAL_TOKEN_COLUMNS = {
    "erc20_most_sent_token_type",
    "erc20_most_rec_token_type",
}

SIGNED_LOG_COLUMNS = {"total_ether_balance"}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Kaggle column names to stable snake_case names."""
    cleaned = df.copy()
    new_columns: list[str] = []
    seen: dict[str, int] = {}

    for index, column in enumerate(cleaned.columns):
        raw_name = str(column).strip()

        if raw_name == "" or raw_name.lower().startswith("unnamed"):
            name = "row_id"
        elif raw_name == "Index":
            name = "dataset_index"
        elif raw_name == "Address":
            name = "address"
        elif raw_name == "FLAG":
            name = "flag"
        else:
            name = _to_snake_case(raw_name)

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        new_columns.append(name)

    cleaned.columns = new_columns
    return cleaned


def split_metadata_and_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Separate metadata/validation columns from numeric clustering features."""
    metadata_columns = [column for column in ("address", "flag") if column in df.columns]
    metadata = df.loc[:, metadata_columns].copy()

    dropped_columns = [
        column
        for column in [*LEAKAGE_AND_ID_COLUMNS, *CATEGORICAL_TOKEN_COLUMNS]
        if column in df.columns
    ]
    features = df.drop(columns=dropped_columns, errors="ignore").copy()

    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    non_numeric_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        features = features.drop(columns=non_numeric_columns)

    summary = {
        "metadata_columns": metadata_columns,
        "dropped_leakage_or_id_columns": [
            column for column in dropped_columns if column in LEAKAGE_AND_ID_COLUMNS
        ],
        "dropped_categorical_columns": [
            column for column in dropped_columns if column in CATEGORICAL_TOKEN_COLUMNS
        ],
        "dropped_non_numeric_columns": non_numeric_columns,
    }

    return metadata, features, summary


def fill_missing_values(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill ERC20 missing values with zero and any remaining missing values by median."""
    filled = features.copy()
    missing_before = filled.isna().sum()

    erc20_missing_columns = [
        column
        for column in filled.columns
        if column.startswith("erc20_") and filled[column].isna().any()
    ]
    if erc20_missing_columns:
        filled.loc[:, erc20_missing_columns] = filled.loc[:, erc20_missing_columns].fillna(0)

    median_filled_columns: list[str] = []
    for column in filled.columns[filled.isna().any()]:
        median_value = filled[column].median()
        if pd.isna(median_value):
            median_value = 0
        filled[column] = filled[column].fillna(median_value)
        median_filled_columns.append(column)

    summary = {
        "missing_values_before": missing_before[missing_before > 0].to_dict(),
        "erc20_zero_filled_columns": erc20_missing_columns,
        "median_filled_columns": median_filled_columns,
        "missing_values_after": int(filled.isna().sum().sum()),
    }
    return filled, summary


def drop_constant_columns(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove columns that have no variation after missing value handling."""
    unique_counts = features.nunique(dropna=False)
    constant_columns = unique_counts[unique_counts <= 1].index.tolist()
    reduced = features.drop(columns=constant_columns)
    return reduced, {"dropped_constant_columns": constant_columns}


def transform_skewed_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply log transforms while preserving signed balance information."""
    transformed = features.copy()
    signed_log_columns = [
        column for column in SIGNED_LOG_COLUMNS if column in transformed.columns
    ]
    log1p_columns: list[str] = []
    skipped_negative_columns: list[str] = []

    for column in transformed.columns:
        if column in signed_log_columns:
            transformed[column] = np.sign(transformed[column]) * np.log1p(
                np.abs(transformed[column])
            )
            continue

        minimum = transformed[column].min()
        maximum = transformed[column].max()
        if minimum >= 0 and maximum > 1:
            transformed[column] = np.log1p(transformed[column])
            log1p_columns.append(column)
        elif minimum < 0:
            skipped_negative_columns.append(column)

    summary = {
        "log1p_transformed_columns": log1p_columns,
        "signed_log_transformed_columns": signed_log_columns,
        "skipped_negative_columns": skipped_negative_columns,
    }
    return transformed, summary


def scale_features(features: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    """Scale clustering features with RobustScaler."""
    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(features)
    scaled = pd.DataFrame(scaled_values, columns=features.columns, index=features.index)
    return scaled, scaler


def run_preprocessing(
    raw_path: str | Path = "data/raw/transaction_dataset.csv",
    interim_dir: str | Path = "data/interim",
    preprocessed_dir: str | Path = "data/preprocessed",
) -> dict[str, Any]:
    """Run the full preprocessing pipeline and save clustering-ready outputs."""
    raw_path = Path(raw_path)
    interim_dir = Path(interim_dir)
    preprocessed_dir = Path(preprocessed_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    interim_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    cleaned = clean_column_names(raw)
    metadata, features, split_summary = split_metadata_and_features(cleaned)
    features, missing_summary = fill_missing_values(features)
    features, constant_summary = drop_constant_columns(features)
    features_clean, transform_summary = transform_skewed_features(features)
    features_scaled, _ = scale_features(features_clean)

    _validate_outputs(features_clean, features_scaled, metadata)

    features_clean.to_csv(interim_dir / "features_clean.csv", index=False)
    metadata.to_csv(interim_dir / "wallet_metadata.csv", index=False)
    features_scaled.to_csv(preprocessed_dir / "features_scaled.csv", index=False)

    summary = {
        "raw_path": str(raw_path),
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "processed_rows": len(features_scaled),
        "processed_feature_columns": len(features_scaled.columns),
        **split_summary,
        **missing_summary,
        **constant_summary,
        **transform_summary,
    }
    _write_summary(summary, interim_dir / "preprocessing_summary.csv")
    return summary


def _to_snake_case(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("tnx", "tnx")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _validate_outputs(
    features_clean: pd.DataFrame,
    features_scaled: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    forbidden = LEAKAGE_AND_ID_COLUMNS | CATEGORICAL_TOKEN_COLUMNS
    leaked_columns = sorted(forbidden.intersection(features_clean.columns))
    if leaked_columns:
        raise ValueError(f"Forbidden columns found in features: {leaked_columns}")

    if len(features_clean) != len(features_scaled) or len(features_clean) != len(metadata):
        raise ValueError("Feature and metadata row counts do not match.")

    if features_clean.isna().any().any() or features_scaled.isna().any().any():
        raise ValueError("Missing values remain after preprocessing.")

    non_numeric_columns = features_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Non-numeric feature columns remain: {non_numeric_columns}")


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, str]] = []
    for key, value in summary.items():
        if isinstance(value, dict):
            formatted_value = "; ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, list):
            formatted_value = "; ".join(map(str, value))
        else:
            formatted_value = str(value)
        rows.append({"item": key, "value": formatted_value})

    pd.DataFrame(rows).to_csv(path, index=False)
