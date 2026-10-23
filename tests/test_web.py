"""Unit test verifying that the exported model bundle loads and produces predictions."""

from pathlib import Path
import pickle
import numpy as np
import pytest


def test_model_bundle_exists_and_predicts():
    """Verify artifacts/model_bundle.pkl is valid and performs inference correctly."""
    bundle_path = Path("artifacts/model_bundle.pkl")
    assert bundle_path.is_file(), "model_bundle.pkl not found"

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    assert "scaler" in bundle
    assert "logistic_regression" in bundle
    assert "perceptron" in bundle
    assert "train_median_ag_ratio" in bundle
    assert "feature_names" in bundle

    # Test sample: 45 yo Male, normal values
    raw_sample = np.array([[45, 1, 1.0, 0.3, 180, 25, 30, 7.0, 3.5, 1.0]])
    scaled_sample = bundle["scaler"].transform(raw_sample)

    prob = bundle["logistic_regression"].predict_proba(scaled_sample)[0]
    assert 0.0 <= prob <= 1.0

    perc_pred = bundle["perceptron"].predict(scaled_sample)[0]
    assert perc_pred in (0, 1)
