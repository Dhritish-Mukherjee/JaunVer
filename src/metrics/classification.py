"""From-scratch classification metrics and confusion matrix computations.

Implements standard binary evaluation metrics without using scikit-learn.
"""

from typing import Dict, Tuple, Union
import numpy as np


def confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> np.ndarray:
    """Compute 2x2 confusion matrix to evaluate binary classification accuracy.

    Layout convention:
        Matrix shape (2, 2):
        [[ TN,  FP ],
         [ FN,  TP ]]

        Row 0: True Negative class (0)
        Row 1: True Positive class (1)
        Col 0: Predicted Negative class (0)
        Col 1: Predicted Positive class (1)

    Parameters
    ----------
    y_true : array-like of shape (m,)
        Ground truth binary labels in {0, 1}.
    y_pred : array-like of shape (m,)
        Estimated target values in {0, 1}.

    Returns
    -------
    np.ndarray of shape (2, 2)
        Confusion matrix.
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()

    if len(yt) != len(yp):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(yt)} vs len(y_pred)={len(yp)}"
        )

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def extract_confusion_counts(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Tuple[int, int, int, int]:
    """Extract scalar counts (TP, FP, TN, FN) directly.

    Returns
    -------
    Tuple[int, int, int, int]
        tp, fp, tn, fn
    """
    cm = confusion_matrix(y_true, y_pred)
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    return tp, fp, tn, fn


def accuracy_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Calculate accuracy: fraction of correctly classified instances.

    Formula: (TP + TN) / (TP + FP + TN + FN)
    """
    tp, fp, tn, fn = extract_confusion_counts(y_true, y_pred)
    total = tp + fp + tn + fn
    return float((tp + tn) / total) if total > 0 else 0.0


def precision_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    zero_division: float = 0.0,
) -> float:
    """Calculate precision: ability of classifier not to label negative as positive.

    Formula: TP / (TP + FP)
    """
    tp, fp, _, _ = extract_confusion_counts(y_true, y_pred)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else zero_division


def recall_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    zero_division: float = 0.0,
) -> float:
    """Calculate recall (sensitivity): ability of classifier to find all positive instances.

    Formula: TP / (TP + FN)
    """
    tp, _, _, fn = extract_confusion_counts(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else zero_division


def specificity_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    zero_division: float = 0.0,
) -> float:
    """Calculate specificity (True Negative Rate): ability to correctly identify healthy cases.

    Formula: TN / (TN + FP)
    """
    _, fp, tn, _ = extract_confusion_counts(y_true, y_pred)
    denom = tn + fp
    return float(tn / denom) if denom > 0 else zero_division


def f1_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    zero_division: float = 0.0,
) -> float:
    """Calculate F1-Score: harmonic mean of precision and recall.

    Formula: 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision_score(y_true, y_pred, zero_division=zero_division)
    r = recall_score(y_true, y_pred, zero_division=zero_division)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0.0 else zero_division


def classification_report(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Generate comprehensive classification metrics dictionary."""
    tp, fp, tn, fn = extract_confusion_counts(y_true, y_pred)
    return {
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
