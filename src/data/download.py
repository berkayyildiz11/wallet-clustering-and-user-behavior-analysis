from __future__ import annotations

from pathlib import Path


DATASET_SLUG = "vagifa/ethereum-frauddetection-dataset"
RAW_DATA_DIR = Path("data/raw")
EXPECTED_DATASET_FILE = RAW_DATA_DIR / "transaction_dataset.csv"


def download_dataset(force: bool = False) -> Path:
    """Download and unzip the Kaggle dataset into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXPECTED_DATASET_FILE.exists() and not force:
        print(f"Dataset already exists: {EXPECTED_DATASET_FILE}")
        return EXPECTED_DATASET_FILE

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        DATASET_SLUG,
        path=RAW_DATA_DIR,
        unzip=True,
        quiet=False,
        force=force,
    )

    if not EXPECTED_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Download finished, but expected file was not found: {EXPECTED_DATASET_FILE}"
        )

    print(f"Dataset downloaded to: {EXPECTED_DATASET_FILE}")
    return EXPECTED_DATASET_FILE


if __name__ == "__main__":
    download_dataset()
