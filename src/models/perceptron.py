"""From-scratch implementation of the classical Rosenblatt Perceptron algorithm.

Linear binary classifier based on error-driven online weight adjustments.
"""

from typing import List, Optional
import numpy as np


class Perceptron:
    """Rosenblatt Perceptron binary classifier.

    Mathematical formulation:
        Linear score: z = w^T x + b
        Activation (Step function):
            y_hat = 1 if z >= 0 else 0
        Update rule for misclassified instance (x_i, y_i):
            w <- w + eta * (y_i - y_hat_i) * x_i
            b <- b + eta * (y_i - y_hat_i)

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate eta (between 0.0 and 1.0).
    n_iterations : int, default=100
        Number of passes (epochs) over the training dataset.
    random_state : Optional[int], default=42
        Seed for reproducible weight initialization.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and preserve existing weights/bias.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 100,
        random_state: Optional[int] = 42,
        warm_start: bool = False,
    ) -> None:
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.random_state: Optional[int] = random_state
        self.warm_start: bool = warm_start

        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.errors_history_: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Fit Perceptron weights using the perceptron mistake-driven learning rule.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Training feature matrix.
        y : np.ndarray of shape (m,)
            Binary target vector in {0, 1}.

        Returns
        -------
        self : Perceptron
            Fitted classifier instance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()

        m, d = X_arr.shape

        if not self.warm_start or self.weights_ is None:
            if self.random_state is not None:
                rng = np.random.default_rng(seed=self.random_state)
                self.weights_ = rng.normal(loc=0.0, scale=0.01, size=d)
            else:
                self.weights_ = np.zeros(d, dtype=np.float64)

            self.bias_ = 0.0
            self.errors_history_ = []

        for _ in range(self.n_iterations):
            epoch_errors = 0
            for xi, target in zip(X_arr, y_arr):
                linear_output = np.dot(xi, self.weights_) + self.bias_
                prediction = 1 if linear_output >= 0.0 else 0
                error = target - prediction

                if error != 0:
                    update = self.learning_rate * error
                    self.weights_ += update * xi
                    self.bias_ += update
                    epoch_errors += 1

            self.errors_history_.append(epoch_errors)

            # Early stopping if linearly separable and 0 errors achieved
            if epoch_errors == 0:
                break

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate the net linear input (w^T x + b).

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input features.

        Returns
        -------
        np.ndarray of shape (m,)
            Net real-valued decision scores.
        """
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Perceptron is not fitted. Call fit() first.")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.weights_) + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels {0, 1} using the Heaviside step function.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input features.

        Returns
        -------
        np.ndarray of shape (m,)
            Predicted binary class labels in {0, 1}.
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, 0)
