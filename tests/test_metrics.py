"""Unit tests for from-scratch classification metrics and confusion matrix."""

import numpy as np
import pytest

from src.metrics.classification import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    extract_confusion_counts,
    f1_score,
    precision_score,
    recall_score,
    specificity_score,
)


def test_hand_worked_metrics_example():
    """Verify metrics against exact hand-worked numbers.

    Hand calculation:
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 1, 1]

    TP = 3 (true 1, pred 1)
    FN = 1 (true 1, pred 0)
    TN = 2 (true 0, pred 0)
    FP = 2 (true 0, pred 1)

    Accuracy    = (3 + 2) / 8 = 5/8 = 0.625
    Precision   = 3 / (3 + 2) = 3/5 = 0.600
    Recall      = 3 / (3 + 1) = 3/4 = 0.750
    Specificity = 2 / (2 + 2) = 2/4 = 0.500
    F1-Score    = 2 * (0.6 * 0.75) / (0.6 + 0.75) = 0.90 / 1.35 = 2/3 ~ 0.6666667
    """
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 1, 1]

    tp, fp, tn, fn = extract_confusion_counts(y_true, y_pred)
    assert (tp, fp, tn, fn) == (3, 2, 2, 1)

    cm = confusion_matrix(y_true, y_pred)
    expected_cm = np.array([
        [2, 2],  # TN, FP
        [1, 3],  # FN, TP
    ])
    assert np.array_equal(cm, expected_cm)

    assert np.isclose(accuracy_score(y_true, y_pred), 0.625)
    assert np.isclose(precision_score(y_true, y_pred), 0.600)
    assert np.isclose(recall_score(y_true, y_pred), 0.750)
    assert np.isclose(specificity_score(y_true, y_pred), 0.500)
    assert np.isclose(f1_score(y_true, y_pred), 2.0 / 3.0)


def test_perfect_classifier():
    """Verify metrics for 100% accurate predictions."""
    y = [1, 0, 1, 0, 1, 0]
    report = classification_report(y, y)
    assert report["accuracy"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["specificity"] == 1.0
    assert report["f1_score"] == 1.0
    assert report["false_positives"] == 0.0
    assert report["false_negatives"] == 0.0


def test_zero_division_safety():
    """Verify that zero division does not raise exceptions and returns default zero."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    # No positives predicted, so precision denom is 0
    prec = precision_score(y_true, y_pred, zero_division=0.0)
    assert prec == 0.0

    # No positive instances exist, so recall denom is 0
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    assert rec == 0.0


def test_length_mismatch_raises():
    """Verify ValueError is raised when prediction and ground truth lengths differ."""
    with pytest.raises(ValueError):
        confusion_matrix([1, 0], [1])
