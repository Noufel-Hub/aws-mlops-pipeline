"""
pipeline/preprocessing.py
--------------------------
SageMaker Processing step: validates raw data, engineers features,
and splits into train / validation sets. Output artefacts are written
to S3 via the SageMaker-managed /opt/ml/processing/output path.

Usage (local test):
    python preprocessing.py \
        --input-data /opt/ml/processing/input \
        --output-train /opt/ml/processing/output/train \
        --output-val /opt/ml/processing/output/validation
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = ["feature_1", "feature_2", "feature_3", "feature_4", "target"]
TARGET_COLUMN = "target"
VAL_SIZE = 0.2
RANDOM_STATE = 42


# ── Validation ───────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing or data is empty."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    logger.info("Schema validation passed. Shape: %s", df.shape)


def validate_nulls(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Drop rows with nulls; raise if null rate exceeds threshold."""
    null_rate = df.isnull().mean()
    high_null = null_rate[null_rate > threshold]
    if not high_null.empty:
        raise ValueError(f"Columns exceed null threshold ({threshold}): {high_null.to_dict()}")
    before = len(df)
    df = df.dropna()
    logger.info("Dropped %d rows with nulls. Remaining: %d", before - len(df), len(df))
    return df


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature transformations.
    Extend this function with domain-specific logic as needed.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    # Example: interaction term between first two features
    if "feature_1" in df.columns and "feature_2" in df.columns:
        df["feature_1x2"] = df["feature_1"] * df["feature_2"]
        logger.info("Created interaction feature: feature_1x2")

    # Standard scaling on all numeric feature columns
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info("Applied StandardScaler to %d feature columns.", len(feature_cols))

    return df


# ── Split & Save ─────────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame, train_path: str, val_path: str) -> None:
    """Split into train/val and write CSV artefacts."""
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=None
    )
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    train_out = os.path.join(train_path, "train.csv")
    val_out = os.path.join(val_path, "validation.csv")

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    logger.info("Train set: %d rows → %s", len(train_df), train_out)
    logger.info("Validation set: %d rows → %s", len(val_df), val_out)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps preprocessing step")
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-train", type=str, default="/opt/ml/processing/output/train")
    parser.add_argument("--output-val", type=str, default="/opt/ml/processing/output/validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load all CSV files from input directory
    csv_files = [f for f in os.listdir(args.input_data) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.input_data}")

    dfs = [pd.read_csv(os.path.join(args.input_data, f)) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d file(s), total rows: %d", len(csv_files), len(df))

    # Pipeline
    validate_schema(df)
    df = validate_nulls(df)
    df = engineer_features(df)
    split_and_save(df, args.output_train, args.output_val)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
