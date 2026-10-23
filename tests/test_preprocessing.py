"""Unit tests for from-scratch preprocessing, scaling, and data splitting."""

import numpy as np
import pytest

from src.data.loader import load_raw_data
from src.preprocessing.pipeline import clean_and_preprocess_data, train_test_split
from src.preprocessing.scaler import StandardScaler


def test_standard_scaler_hand_worked_numerical_example():
    """Verify StandardScaler against exact hand-calculated values.

    Hand-worked numbers:
    X_train = [[10], [20], [30], [40]]
    mu = 25.0
    var = (225 + 25 + 25 + 225) / 4 = 125.0
    sigma = sqrt(125.0) ~ 11.180339887
    z = (X - 25.0) / 11.180339887
    """
    X_train = np.array([[10.0], [20.0], [30.0], [40.0]])
    scaler = StandardScaler()
    scaler.fit(X_train)

    assert np.isclose(scaler.mean_[0], 25.0)
    assert np.isclose(scaler.std_[0], np.sqrt(125.0))

    X_train_scaled = scaler.transform(X_train)
    expected_train_scaled = np.array([
        [-1.34164079],
        [-0.44721360],
        [0.44721360],
        [1.34164079],
    ])
    assert np.allclose(X_train_scaled, expected_train_scaled, atol=1e-6)

    # Test point at 25.0 must scale to exactly 0.0
    X_test = np.array([[25.0]])
    X_test_scaled = scaler.transform(X_test)
    assert np.isclose(X_test_scaled[0, 0], 0.0)


def test_standard_scaler_raises_when_not_fitted():
    """Verify that transform raises RuntimeError if called prior to fit."""
    scaler = StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_train_test_split_shapes_and_reproducibility():
    """Verify manual train_test_split proportions, no overlap, and seed reproducibility."""
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)

    # 80/20 split
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    assert len(X_tr1) == 80
    assert len(X_te1) == 20
    assert len(y_tr1) == 80
    assert len(y_te1) == 20

    # Ensure no data overlap between train and test
    overlap = set(y_tr1).intersection(set(y_te1))
    assert len(overlap) == 0

    # Same seed must yield identical results
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    assert np.array_equal(y_tr1, y_tr2)

    # Different seed must yield different split
    _, _, _, y_te3 = train_test_split(X, y, test_size=0.2, random_state=99)
    assert not np.array_equal(y_te1, y_te3)


def test_clean_and_preprocess_pipeline_end_to_end():
    """Verify end-to-end preprocessing pipeline on the real ILPD dataset."""
    df = load_raw_data("data/raw/indian_liver_patient.csv")
    prep = clean_and_preprocess_data(df, test_size=0.2, random_state=42)

    # 13 duplicates dropped: 583 - 13 = 570
    assert prep["duplicates_dropped"] == 13

    # 570 rows -> 80% train = 456, 20% test = 114
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    assert X_train.shape == (456, 10)
    assert X_test.shape == (114, 10)
    assert y_train.shape == (456,)
    assert y_test.shape == (114,)

    # Zero missing values remain
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # Targets strictly binary {0, 1}
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})

    # Train features must have mean ~ 0 and std ~ 1
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-7)
