"""From-scratch Logistic Regression classifier using Batch Gradient Descent.

Implements binary cross-entropy optimization without external machine learning libraries.
"""

from typing import List, Optional, Union
import numpy as np


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute the numerically stable Sigmoid activation function.

    Formula:
        sigma(z) = 1 / (1 + exp(-z))

    Parameters
    ----------
    z : float or np.ndarray
        Linear logit values.

    Returns
    -------
    float or np.ndarray
        Activated probability values in the open interval (0, 1).
    """
    # Clip z to prevent floating-point overflow in exp(-z)
    z_clipped = np.clip(z, -250.0, 250.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute Binary Cross-Entropy (Log Loss) with numerical stabilization.

    Formula:
        J(w, b) = - (1 / m) * sum( y * log(y_pred) + (1 - y) * log(1 - y_pred) )

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels of shape (m,), with values in {0, 1}.
    y_pred : np.ndarray
        Predicted probabilities of shape (m,), with values in (0, 1).
    eps : float, default=1e-15
        Clipping epsilon to prevent log(0) undefined behavior.

    Returns
    -------
    float
        Scalar loss value.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64).ravel(), eps, 1.0 - eps)
    m = len(y_t)
    if m == 0:
        return 0.0
    loss = - (1.0 / m) * np.sum(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
    return float(loss)


class LogisticRegression:
    """Binary Logistic Regression with Batch Gradient Descent.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size (alpha) used in gradient descent weight updates.
    n_iterations : int, default=1000
        Total number of gradient descent iterations (epochs).
    random_state : Optional[int], default=42
        Seed for reproducible weight initialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        random_state: Optional[int] = 42,
    ) -> None:
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.random_state: Optional[int] = random_state

        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit model parameters using batch gradient descent.

        Optimization formulation:
            z = X @ w + b
            y_hat = sigmoid(z)
            dw = (1 / m) * X.T @ (y_hat - y)
            db = (1 / m) * sum(y_hat - y)
            w = w - learning_rate * dw
            b = b - learning_rate * db

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Training feature matrix.
        y : np.ndarray of shape (m,)
            Target labels in {0, 1}.

        Returns
        -------
        self : LogisticRegression
            Fitted classifier instance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        m, d = X_arr.shape

        # Initialize weights to zeros (or small random numbers if seed provided)
        if self.random_state is not None:
            rng = np.random.default_rng(seed=self.random_state)
            self.weights_ = rng.normal(loc=0.0, scale=0.01, size=d)
        else:
            self.weights_ = np.zeros(d, dtype=np.float64)

        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            # 1. Forward Pass
            linear_output = np.dot(X_arr, self.weights_) + self.bias_
            y_hat = sigmoid(linear_output)

            # 2. Record Loss
            loss = binary_cross_entropy(y_arr, y_hat)
            self.loss_history_.append(loss)

            # 3. Compute Gradients
            error = y_hat - y_arr
            dw = (1.0 / m) * np.dot(X_arr.T, error)
            db = float((1.0 / m) * np.sum(error))

            # 4. Gradient Descent Update
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate posterior probability of the positive class P(Y=1 | X).

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input features.

        Returns
        -------
        np.ndarray of shape (m,)
            Predicted probabilities for class 1.
        """
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")

        X_arr = np.asarray(X, dtype=np.float64)
        linear_output = np.dot(X_arr, self.weights_) + self.bias_
        return sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels {0, 1} for input samples.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input features.
        threshold : float, default=0.5
            Decision boundary probability threshold.

        Returns
        -------
        np.ndarray of shape (m,)
            Predicted labels in {0, 1}.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
