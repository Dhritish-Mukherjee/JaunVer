"""Unit tests for from-scratch K-Nearest Neighbors (KNN) classifier."""

import numpy as np
import pytest

from src.data.loader import load_raw_data
from src.models.knn import KNearestNeighbors
from src.preprocessing.pipeline import clean_and_preprocess_data


def test_knn_hand_worked_numerical_example():
    """Verify KNN distance and majority voting against hand-calculated values.

    Training data:
    Point 0: [0, 0] -> Label 0
    Point 1: [1, 1] -> Label 0
    Point 2: [5, 5] -> Label 1
    Point 3: [6, 6] -> Label 1

    Query A: [0.5, 0.5]
    Distances:
      d0 = sqrt(0.5^2 + 0.5^2) = sqrt(0.50) ~ 0.7071
      d1 = sqrt(0.5^2 + 0.5^2) = sqrt(0.50) ~ 0.7071
      d2 = sqrt(4.5^2 + 4.5^2) = sqrt(40.5) ~ 6.3639
      d3 = sqrt(5.5^2 + 5.5^2) = sqrt(60.5) ~ 7.7781

    For k=3: 3 closest are {0, 1, 2} with labels [0, 0, 1].
    Majority vote: 0
    P(Y=1) = 1/3 ~ 0.3333
    """
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    y_train = np.array([0, 0, 1, 1])

    knn = KNearestNeighbors(n_neighbors=3)
    knn.fit(X_train, y_train)

    query_a = np.array([[0.5, 0.5]])
    dists, indices = knn.kneighbors(query_a)

    assert set(indices[0]) == {0, 1, 2}
    assert np.isclose(dists[0, 0], np.sqrt(0.5))
    assert np.isclose(dists[0, 1], np.sqrt(0.5))

    prob_a = knn.predict_proba(query_a)[0]
    assert np.isclose(prob_a, 1.0 / 3.0)
    assert knn.predict(query_a)[0] == 0

    # Query B: near class 1
    query_b = np.array([[5.5, 5.5]])
    prob_b = knn.predict_proba(query_b)[0]
    assert np.isclose(prob_b, 2.0 / 3.0)
    assert knn.predict(query_b)[0] == 1


def test_knn_invalid_parameters():
    """Verify validation of k neighbors parameter."""
    with pytest.raises(ValueError):
        KNearestNeighbors(n_neighbors=0)

    with pytest.raises(ValueError):
        KNearestNeighbors(n_neighbors=-5)

    knn = KNearestNeighbors(n_neighbors=10)
    with pytest.raises(ValueError):
        knn.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))


def test_knn_not_fitted_raises():
    """Verify RuntimeError is raised when predicting before fit."""
    knn = KNearestNeighbors(n_neighbors=3)
    with pytest.raises(RuntimeError):
        knn.predict(np.array([[1.0, 2.0]]))


def test_knn_fit_predict_on_liver_dataset():
    """Verify KNN trains and evaluates smoothly on the ILPD liver dataset."""
    df = load_raw_data("data/raw/indian_liver_patient.csv")
    prep = clean_and_preprocess_data(df, test_size=0.2, random_state=42)

    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]

    knn = KNearestNeighbors(n_neighbors=5)
    knn.fit(X_train, y_train)

    probs = knn.predict_proba(X_test)
    preds = knn.predict(X_test)

    assert probs.shape == (len(y_test),)
    assert preds.shape == (len(y_test),)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert set(np.unique(preds)).issubset({0, 1})
