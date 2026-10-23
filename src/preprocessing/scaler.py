"""From-scratch implementation of feature standardization (Z-score normalization).

Standardization scales feature vectors so that each feature has a mean of 0
and a standard deviation of 1.
"""

from typing import Optional, Union
import numpy as np


class StandardScaler:
    """Standardize features by centering to mean 0 and scaling to unit variance.

    Mathematical formulation for feature column j:
        mu_j = (1 / m) * sum(x_{i,j})
        sigma_j = sqrt((1 / m) * sum((x_{i,j} - mu_j)^2))
        z_{i,j} = (x_{i,j} - mu_j) / sigma_j

    Parameters
    ----------
    eps : float, default=1e-8
        A small positive constant added to prevent division by zero in case of
        constant features with zero standard deviation.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps: float = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, list]) -> "StandardScaler":
        """Compute the mean and standard deviation from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data used to compute scaling parameters.

        Returns
        -------
        self : StandardScaler
            Fitted scaler instance.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        self.mean_ = np.mean(arr, axis=0)
        self.std_ = np.std(arr, axis=0)
        return self

    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Standardize feature matrix using the fitted mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Standardized feature matrix of shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError(
                "StandardScaler has not been fitted yet. Call fit() before transform()."
            )

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Protect against division by zero if variance is 0
        safe_std = np.where(self.std_ < self.eps, self.eps, self.std_)
        return (arr - self.mean_) / safe_std

    def fit_transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Fit to training data, then transform it in a single pass.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray
            Standardized feature matrix.
        """
        return self.fit(X).transform(X)
