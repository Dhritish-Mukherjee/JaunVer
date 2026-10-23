"""Unit tests verifying mathematical correctness of Logistic Regression and Perceptron."""

import numpy as np
import pytest

from src.data.loader import load_raw_data
from src.models.logistic_regression import (
    LogisticRegression,
    binary_cross_entropy,
    sigmoid,
)
from src.models.perceptron import Perceptron
from src.preprocessing.pipeline import clean_and_preprocess_data


def test_sigmoid_values_and_numerical_stability():
    """Verify sigmoid values at boundary points and under extreme numerical values."""
    assert np.isclose(sigmoid(0.0), 0.5)
    assert sigmoid(100.0) > 0.9999
    assert sigmoid(-100.0) < 0.0001

    # Should not raise OverflowError with large numbers
    large_pos = sigmoid(1000.0)
    large_neg = sigmoid(-1000.0)
    assert np.isclose(large_pos, 1.0)
    assert np.isclose(large_neg, 0.0)


def test_binary_cross_entropy_exact_values():
    """Verify binary cross-entropy loss against known analytical values."""
    # When predictions are completely uninformative (0.5 for all classes),
    # BCE = -log(0.5) ~ 0.693147
    y_true = np.array([1, 0])
    y_pred = np.array([0.5, 0.5])
    expected_bce = -np.log(0.5)
    assert np.isclose(binary_cross_entropy(y_true, y_pred), expected_bce, atol=1e-5)


def test_logistic_regression_hand_worked_single_step():
    """Verify single gradient descent step against exact manual derivation.

    Hand-worked numbers:
    m = 1, X = [[1.0, 2.0]], y = [1.0]
    initial w = [0.0, 0.0], b = 0.0, lr = 0.1
    z = 0.0, y_hat = 0.5
    error = -0.5
    dw = [-0.5, -1.0], db = -0.5
    new w = [0.05, 0.10], new b = 0.05
    """
    X = np.array([[1.0, 2.0]])
    y = np.array([1])

    clf = LogisticRegression(learning_rate=0.1, n_iterations=1, random_state=None)
    clf.fit(X, y)

    assert np.allclose(clf.weights_, [0.05, 0.10], atol=1e-6)
    assert np.isclose(clf.bias_, 0.05, atol=1e-6)


def test_logistic_regression_loss_strictly_decreases_on_separable_data():
    """Verify that batch gradient descent minimizes BCE loss monotonically."""
    X = np.array([
        [-2.0, -2.0],
        [-1.0, -1.5],
        [1.5, 2.0],
        [2.0, 2.5],
    ])
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.5, n_iterations=100, random_state=42)
    clf.fit(X, y)

    # Initial loss should be higher than final loss
    assert clf.loss_history_[0] > clf.loss_history_[-1]
    # Check predictions
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_perceptron_hand_worked_update_step():
    """Verify Perceptron update rule against manual step calculation.

    Initial w = [0.0, 0.0], b = -1.0, eta = 0.1
    Sample: x = [1.0, -2.0], true y = 1
    Score: z = 0 - 1.0 = -1.0 < 0 => y_hat = 0
    Error: y - y_hat = 1 - 0 = 1
    Update:
        w <- w + 0.1 * 1 * [1.0, -2.0] = [0.1, -0.2]
        b <- -1.0 + 0.1 * 1 = -0.9
    """
    p = Perceptron(learning_rate=0.1, n_iterations=1, random_state=None, warm_start=True)
    # Manually configure state to test exact update
    p.weights_ = np.array([0.0, 0.0])
    p.bias_ = -1.0

    X = np.array([[1.0, -2.0]])
    y = np.array([1])

    # Perform 1 fit epoch
    p.fit(X, y)

    assert np.allclose(p.weights_, [0.1, -0.2], atol=1e-6)
    assert np.isclose(p.bias_, -0.9, atol=1e-6)



def test_models_fit_and_predict_on_liver_dataset():
    """Verify that both models train and produce valid binary predictions on real data."""
    df = load_raw_data("data/raw/indian_liver_patient.csv")
    prep = clean_and_preprocess_data(df, test_size=0.2, random_state=42)

    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]

    # Logistic Regression
    lr = LogisticRegression(learning_rate=0.1, n_iterations=200, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    assert lr_preds.shape == (len(y_test),)
    assert set(np.unique(lr_preds)).issubset({0, 1})
    assert len(lr.loss_history_) == 200

    # Perceptron
    perc = Perceptron(learning_rate=0.01, n_iterations=100, random_state=42)
    perc.fit(X_train, y_train)
    perc_preds = perc.predict(X_test)
    assert perc_preds.shape == (len(y_test),)
    assert set(np.unique(perc_preds)).issubset({0, 1})
