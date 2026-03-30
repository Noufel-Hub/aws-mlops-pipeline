"""
pipeline/training.py
---------------------
SageMaker Training step: loads preprocessed train/validation data from S3,
trains a scikit-learn model, and writes the model artefact to /opt/ml/model
for automatic upload to S3 by SageMaker.

SageMaker injects these environment variables automatically:
    SM_CHANNEL_TRAIN   → path to training data
    SM_CHANNEL_VALIDATION → path to validation data
    SM_MODEL_DIR       → path to write the model artefact
    SM_OUTPUT_DATA_DIR → path for any additional outputs (e.g. metrics)
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, filename: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load CSV and return features (X) and labels (y)."""
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN]).values
    logger.info("Loaded %s — X: %s, y: %s", path, X.shape, y.shape)
    return X, y


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(n_estimators: int, max_depth: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    preds = model.predict(X)
    return {
        "accuracy": round(accuracy_score(y, preds), 4),
        "f1_weighted": round(f1_score(y, preds, average="weighted"), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps training step")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    # SageMaker-injected paths
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_train, y_train = load_dataset(args.train, "train.csv")
    X_val, y_val = load_dataset(args.validation, "validation.csv")

    logger.info("Training RandomForest — n_estimators=%d, max_depth=%d", args.n_estimators, args.max_depth)
    model = build_model(args.n_estimators, args.max_depth)
    model.fit(X_train, y_train)

    train_metrics = compute_metrics(model, X_train, y_train)
    val_metrics = compute_metrics(model, X_val, y_val)

    logger.info("Train metrics: %s", train_metrics)
    logger.info("Validation metrics: %s", val_metrics)

    # Write metrics for the evaluation step to consume
    os.makedirs(args.output_data_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_data_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"train": train_metrics, "validation": val_metrics}, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save model artefact — SageMaker uploads /opt/ml/model to S3 automatically
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
