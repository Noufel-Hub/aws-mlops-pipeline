"""
pipeline/evaluation.py
-----------------------
SageMaker Processing step: loads the trained model and validation data,
computes evaluation metrics, and writes a report used by the pipeline
condition step to decide whether to register the model or abort.
 
If validation F1 >= APPROVAL_THRESHOLD → model is approved for registry.
If validation F1 <  APPROVAL_THRESHOLD → pipeline halts, model is rejected.
"""
 
import argparse
import json
import logging
import os
import tarfile
 
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
TARGET_COLUMN = "target"
APPROVAL_THRESHOLD = 0.80   # Minimum F1 (weighted) required to register the model
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def load_model(model_dir: str):
    """Extract model.tar.gz (SageMaker format) and load the artefact."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        logger.info("Extracting %s", tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(model_dir)
 
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artefact not found at {model_path}")
 
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model
 
 
def load_validation_data(val_dir: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(val_dir, "validation.csv")
    df = pd.read_csv(path)
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN]).values
    logger.info("Validation data loaded — X: %s, y: %s", X.shape, y.shape)
    return X, y
 
 
# ── Evaluation ───────────────────────────────────────────────────────────────
 
def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute full evaluation report."""
    preds = model.predict(X)
 
    metrics = {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "f1_weighted": round(float(f1_score(y, preds, average="weighted")), 4),
        "f1_macro": round(float(f1_score(y, preds, average="macro")), 4),
    }
 
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
 
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Confusion matrix:\n%s", confusion_matrix(y, preds))
 
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "approval_threshold": APPROVAL_THRESHOLD,
        "approved": metrics["f1_weighted"] >= APPROVAL_THRESHOLD,
    }
 
 
# ── Main ─────────────────────────────────────────────────────────────────────
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps evaluation step")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--validation-dir", type=str, default="/opt/ml/processing/validation")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
 
    model = load_model(args.model_dir)
    X_val, y_val = load_validation_data(args.validation_dir)
    report = evaluate(model, X_val, y_val)
 
    # Write evaluation report — SageMaker pipeline condition step reads this
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report written to %s", report_path)
 
    if report["approved"]:
        logger.info(
            "✅ Model APPROVED (F1=%.4f >= threshold=%.2f) — proceeding to Model Registry.",
            report["metrics"]["f1_weighted"],
            APPROVAL_THRESHOLD,
        )
    else:
        logger.warning(
            "❌ Model REJECTED (F1=%.4f < threshold=%.2f) — pipeline will halt.",
            report["metrics"]["f1_weighted"],
            APPROVAL_THRESHOLD,
        )
 
 
if __name__ == "__main__":
    main()
 
