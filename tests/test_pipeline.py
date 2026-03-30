"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for the MLOps pipeline steps.

Tests cover:
  - preprocessing: schema validation, null handling, feature engineering, output shape
  - training:      model training, metric computation, artefact saving
  - evaluation:    approval threshold logic, report structure

Run with:
    pytest tests/test_pipeline.py -v
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

# Make pipeline/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.preprocessing import engineer_features, validate_nulls, validate_schema
from pipeline.training import build_model, compute_metrics
from pipeline.evaluation import APPROVAL_THRESHOLD, evaluate


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Minimal valid dataframe matching the expected schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randn(n),
        "feature_4": np.random.randn(n),
        "target":    np.random.randint(0, 2, n),
    })


@pytest.fixture
def trained_model(valid_dataframe) -> tuple:
    """Returns a fitted model and the validation arrays."""
    df = valid_dataframe.copy()
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    model = build_model(n_estimators=10, max_depth=3)
    model.fit(X, y)
    return model, X, y


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateSchema:

    def test_passes_with_correct_columns(self, valid_dataframe):
        validate_schema(valid_dataframe)   # Should not raise

    def test_raises_on_missing_column(self, valid_dataframe):
        df = valid_dataframe.drop(columns=["feature_1"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["feature_1", "feature_2", "feature_3", "feature_4", "target"])
        with pytest.raises(ValueError, match="empty"):
            validate_schema(df)


class TestValidateNulls:

    def test_drops_null_rows(self, valid_dataframe):
        df = valid_dataframe.copy()
        df.loc[0, "feature_1"] = None
        result = validate_nulls(df)
        assert len(result) == len(valid_dataframe) - 1

    def test_raises_when_null_rate_exceeds_threshold(self, valid_dataframe):
        df = valid_dataframe.copy()
        # Set 50% of feature_1 to null → exceeds default 10% threshold
        df.loc[:49, "feature_1"] = None
        with pytest.raises(ValueError, match="null threshold"):
            validate_nulls(df)

    def test_passes_clean_dataframe(self, valid_dataframe):
        result = validate_nulls(valid_dataframe)
        assert len(result) == len(valid_dataframe)


class TestEngineerFeatures:

    def test_creates_interaction_feature(self, valid_dataframe):
        result = engineer_features(valid_dataframe.copy())
        assert "feature_1x2" in result.columns

    def test_output_has_no_nulls(self, valid_dataframe):
        result = engineer_features(valid_dataframe.copy())
        assert result.isnull().sum().sum() == 0

    def test_feature_count_increases(self, valid_dataframe):
        original_cols = len(valid_dataframe.columns)
        result = engineer_features(valid_dataframe.copy())
        assert len(result.columns) > original_cols


# ══════════════════════════════════════════════════════════════════════════════
# Training Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildModel:

    def test_returns_random_forest(self):
        model = build_model(n_estimators=10, max_depth=3)
        assert isinstance(model, RandomForestClassifier)

    def test_hyperparameters_set_correctly(self):
        model = build_model(n_estimators=50, max_depth=5)
        assert model.n_estimators == 50
        assert model.max_depth == 5


class TestComputeMetrics:

    def test_returns_expected_keys(self, trained_model):
        model, X, y = trained_model
        metrics = compute_metrics(model, X, y)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics

    def test_metrics_in_valid_range(self, trained_model):
        model, X, y = trained_model
        metrics = compute_metrics(model, X, y)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_weighted"] <= 1.0

    def test_model_artefact_saved(self, trained_model):
        import joblib
        model, _, _ = trained_model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            joblib.dump(model, path)
            assert os.path.exists(path)
            loaded = joblib.load(path)
            assert isinstance(loaded, RandomForestClassifier)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def test_report_structure(self, trained_model):
        model, X, y = trained_model
        report = evaluate(model, X, y)
        assert "metrics" in report
        assert "approved" in report
        assert "confusion_matrix" in report
        assert "approval_threshold" in report

    def test_approval_flag_true_when_above_threshold(self, trained_model):
        """A model that perfectly predicts should be approved."""
        model, X, y = trained_model
        # Force perfect predictions by using training data
        model.fit(X, y)
        report = evaluate(model, X, y)
        # Training accuracy should exceed threshold
        if report["metrics"]["f1_weighted"] >= APPROVAL_THRESHOLD:
            assert report["approved"] is True

    def test_approval_flag_false_when_below_threshold(self, valid_dataframe):
        """A random-label model should be rejected."""
        X = valid_dataframe.drop(columns=["target"]).values
        y_true = valid_dataframe["target"].values
        y_random = np.random.randint(0, 2, len(y_true))

        # Build a mock model that always predicts the majority class (worst case)
        model = build_model(n_estimators=1, max_depth=1)
        model.fit(X, y_true)

        # Patch predict to return random labels for a guaranteed poor score
        model.predict = lambda _: y_random

        report = evaluate(model, X, y_true)
        # If F1 is below threshold, approved must be False
        if report["metrics"]["f1_weighted"] < APPROVAL_THRESHOLD:
            assert report["approved"] is False

    def test_report_is_json_serialisable(self, trained_model):
        model, X, y = trained_model
        report = evaluate(model, X, y)
        # Should not raise
        serialised = json.dumps(report)
        assert isinstance(serialised, str)


# ══════════════════════════════════════════════════════════════════════════════
# Integration Test: Preprocessing → Training → Evaluation
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:

    def test_full_pipeline_runs(self, valid_dataframe):
        """
        Smoke test: runs the full preprocessing → training → evaluation
        flow in memory. Validates that outputs have the expected shape
        and types at each stage.
        """
        # Preprocessing
        df = valid_dataframe.copy()
        validate_schema(df)
        df = validate_nulls(df)
        df = engineer_features(df)
        assert len(df) > 0
        assert "target" in df.columns

        # Training
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        model = build_model(n_estimators=10, max_depth=3)
        model.fit(X, y)
        metrics = compute_metrics(model, X, y)
        assert metrics["accuracy"] > 0

        # Evaluation
        report = evaluate(model, X, y)
        assert isinstance(report["approved"], bool)
        assert report["approval_threshold"] == APPROVAL_THRESHOLD
