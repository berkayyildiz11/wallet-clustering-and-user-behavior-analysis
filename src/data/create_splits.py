from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURES_PATH = Path("data/preprocessed/features_scaled.csv")
OUTPUT_DIR = Path("data/splits")
SPLIT_FILE = OUTPUT_DIR / "split_indices.csv"
SUMMARY_FILE = OUTPUT_DIR / "split_summary.csv"

RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def create_shared_splits(
    features_path: str | Path = FEATURES_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create reproducible train/validation/test row assignments.

    These splits are row-index based and do not use FLAG, address, or cluster labels.
    Clustering can still be fit on all rows; the splits are mainly for downstream
    pseudo-label XGBoost evaluation and shared stability checks.
    """
    features_path = Path(features_path)
    output_dir = Path(output_dir)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Preprocessed features not found: {features_path}. "
            "Run `uv run python -m src.data.run_preprocessing` first."
        )

    features = pd.read_csv(features_path)
    wallet_indices = pd.Series(range(len(features)), name="wallet_index")

    train_indices, temp_indices = train_test_split(
        wallet_indices,
        train_size=TRAIN_SIZE,
        random_state=random_seed,
        shuffle=True,
    )

    validation_ratio_of_temp = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_indices, test_indices = train_test_split(
        temp_indices,
        train_size=validation_ratio_of_temp,
        random_state=random_seed,
        shuffle=True,
    )

    split_df = pd.concat(
        [
            _make_split_frame(train_indices, "train"),
            _make_split_frame(validation_indices, "validation"),
            _make_split_frame(test_indices, "test"),
        ],
        ignore_index=True,
    ).sort_values("wallet_index")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_dir / "split_indices.csv", index=False)
    _write_split_summary(split_df, output_dir / "split_summary.csv", random_seed)

    print(f"Split file written to: {output_dir / 'split_indices.csv'}")
    print(f"Summary written to: {output_dir / 'split_summary.csv'}")
    print(split_df["split"].value_counts().sort_index().to_string())
    return split_df


def _make_split_frame(indices: pd.Series, split_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wallet_index": indices.to_numpy(),
            "split": split_name,
        }
    )


def _write_split_summary(
    split_df: pd.DataFrame,
    summary_path: Path,
    random_seed: int,
) -> None:
    counts = split_df["split"].value_counts().rename_axis("split").reset_index(name="rows")
    counts["fraction"] = counts["rows"] / len(split_df)
    counts["random_seed"] = random_seed
    counts.sort_values("split").to_csv(summary_path, index=False)


if __name__ == "__main__":
    create_shared_splits()
