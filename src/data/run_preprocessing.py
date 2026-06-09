from __future__ import annotations

from src.data.preprocessing import run_preprocessing


def run_preprocessing_pipeline() -> None:
    summary = run_preprocessing(
        raw_path="data/raw/transaction_dataset.csv",
        interim_dir="data/interim",
        preprocessed_dir="data/preprocessed",
    )
    print("Preprocessing complete.")
    print(f"Rows processed: {summary['processed_rows']}")
    print(f"Feature columns: {summary['processed_feature_columns']}")
    print("Interim outputs written to data/interim/")
    print("Final clustering input written to data/preprocessed/")


if __name__ == "__main__":
    run_preprocessing_pipeline()
