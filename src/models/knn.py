"""From-scratch implementation of K-Nearest Neighbors (KNN) classifier.

Non-parametric, instance-based learning algorithm using vectorized Euclidean distance.
"""

from typing import Optional, Tuple, Union
import numpy as np


class KNearestNeighbors:
    """K-Nearest Neighbors binary classifier implemented from scratch using NumPy.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors (k) used for majority voting.
        An odd number is recommended to avoid tie-breaking ambiguity.

    Attributes
    ----------
    X_train_ : np.ndarray of shape (m, d)
        Stored training feature matrix.
    y_train_ : np.ndarray of shape (m,)
        Stored training target labels in {0, 1}.
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
        self.n_neighbors: int = n_neighbors
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        """Store the training feature matrix and target labels.

        As a non-parametric lazy learner, fitting simply stores the training instances.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Standardized training feature matrix.
        y : np.ndarray of shape (m,)
            Binary target labels in {0, 1}.

        Returns
        -------
        self : KNearestNeighbors
            Fitted classifier instance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Length mismatch: len(X)={len(X_arr)} vs len(y)={len(y_arr)}"
            )
        if self.n_neighbors > len(X_arr):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) cannot exceed training samples ({len(X_arr)})"
            )

        self.X_train_ = np.copy(X_arr)
        self.y_train_ = np.copy(y_arr)
        return self

    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between query samples and all training points.

        Formula for distance between query x and training sample t:
            d(x, t) = sqrt( sum_{j=1}^d (x_j - t_j)^2 )
        """
        if self.X_train_ is None:
            raise RuntimeError("Classifier is not fitted. Call fit() before predicting.")

        X_queries = np.asarray(X, dtype=np.float64)
        if X_queries.ndim == 1:
            X_queries = X_queries.reshape(1, -1)

        # Compute Euclidean distance from each query sample to all training samples
        distances = [
            np.sqrt(np.sum((self.X_train_ - q) ** 2, axis=1))
            for q in X_queries
        ]
        return np.array(distances)

    def kneighbors(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the k-nearest neighbors of query points."""
        distances = self._compute_pairwise_distances(X)
        # 1. Indices of the k nearest neighbors
        k_indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        # 2. Sorted distance values to those k neighbors
        k_distances = np.sort(distances, axis=1)[:, : self.n_neighbors]
        return k_distances, k_indices

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate positive class probability P(Y=1 | X) by local neighbor fraction.

        Formula:
            P(Y = 1 | x) = (1 / k) * sum_{i in N_k(x)} y^{(i)}

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, d)
            Query features.

        Returns
        -------
        np.ndarray of shape (n_queries,)
            Estimated posterior probability of class 1.
        """
        _, k_indices = self.kneighbors(X)
        neighbor_labels = self.y_train_[k_indices]
        probabilities = np.mean(neighbor_labels == 1, axis=1)
        return probabilities

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels {0, 1} by majority vote.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, d)
            Query features.
        threshold : float, default=0.5
            Probability cutoff for majority vote.

        Returns
        -------
        np.ndarray of shape (n_queries,)
            Predicted binary labels.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
