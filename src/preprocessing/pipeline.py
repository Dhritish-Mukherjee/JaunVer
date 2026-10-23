"""From-scratch data preprocessing and leakage-free splitting pipeline.

Implements manual train/test splitting, deduplication, categorical encoding,
training-median imputation, and training-only feature standardization.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

from src.data.loader import TARGET_COLUMN
from src.preprocessing.scaler import StandardScaler


def train_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays or matrices into random train and test subsets manually.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        Feature matrix of shape (n_samples, n_features).
    y : Union[np.ndarray, pd.Series]
        Target labels of shape (n_samples,).
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0.0 to 1.0).
    random_state : int, default=42
        Seed used by the random number generator for reproducibility.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test as NumPy arrays.

    Raises
    ------
    ValueError
        If test_size is not between 0 and 1 or if X and y lengths mismatch.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got len(X)={len(X_arr)} and len(y)={len(y_arr)}."
        )

    n_samples = len(X_arr)
    n_test = int(np.round(n_samples * test_size))

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        indices = rng.permutation(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X_arr[train_indices], X_arr[test_indices]
    y_train, y_test = y_arr[train_indices], y_arr[test_indices]

    return X_train, X_test, y_train, y_test


def clean_and_preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Execute the end-to-end preprocessing pipeline with zero data leakage.

    Steps:
    1. Drop exact duplicate rows.
    2. Encode target: 1 (diseased) -> 1, 2 (healthy) -> 0.
    3. Binary-encode 'gender': Male -> 1, Female -> 0.
    4. Split into train and test subsets using a fixed seed.
    5. Impute missing values using the median of the TRAINING split only.
    6. Fit StandardScaler on TRAINING features only, then transform train and test.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame.
    test_size : float, default=0.2
        Fraction of data reserved for testing.
    random_state : int, default=42
        Seed for reproducible splitting.

    Returns
    -------
    Dict[str, object]
        Dictionary containing:
        - 'X_train': np.ndarray, scaled training features
        - 'X_test': np.ndarray, scaled testing features
        - 'y_train': np.ndarray, binary training labels {0, 1}
        - 'y_test': np.ndarray, binary testing labels {0, 1}
        - 'feature_names': List[str], list of predictor names
        - 'scaler': StandardScaler, fitted on training data
        - 'train_median_ag_ratio': float, median used for imputation
        - 'duplicates_dropped': int, number of dropped duplicates
    """
    raw_count = len(df)
    clean_df = df.drop_duplicates().copy()
    duplicates_dropped = raw_count - len(clean_df)

    # 1. Target encoding: 1 (patient) -> 1, 2 (healthy) -> 0
    clean_df[TARGET_COLUMN] = (clean_df[TARGET_COLUMN] == 1).astype(int)

    # 2. Gender encoding: Male -> 1, Female -> 0
    clean_df["gender"] = (
        clean_df["gender"].astype(str).str.strip().str.lower() == "male"
    ).astype(int)

    feature_cols = [c for c in clean_df.columns if c != TARGET_COLUMN]
    X_df = clean_df[feature_cols].copy()
    y_series = clean_df[TARGET_COLUMN].copy()

    # 3. Split data FIRST before computing imputation or scaling statistics
    train_idx, test_idx = _get_split_indices(
        n_samples=len(clean_df),
        test_size=test_size,
        random_state=random_state,
    )

    X_train_df = X_df.iloc[train_idx].copy()
    X_test_df = X_df.iloc[test_idx].copy()
    y_train = y_series.iloc[train_idx].values.astype(int)
    y_test = y_series.iloc[test_idx].values.astype(int)

    # 4. Leakage-free imputation: compute median strictly on training subset
    train_median_ag_ratio = float(
        X_train_df["albumin_and_globulin_ratio"].dropna().median()
    )

    X_train_df["albumin_and_globulin_ratio"] = X_train_df[
        "albumin_and_globulin_ratio"
    ].fillna(train_median_ag_ratio)
    X_test_df["albumin_and_globulin_ratio"] = X_test_df[
        "albumin_and_globulin_ratio"
    ].fillna(train_median_ag_ratio)

    # 5. Leakage-free scaling: fit on X_train only, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "train_median_ag_ratio": train_median_ag_ratio,
        "duplicates_dropped": duplicates_dropped,
    }


def _get_split_indices(
    n_samples: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to generate reproducible train and test index arrays."""
    n_test = int(np.round(n_samples * test_size))
    rng = np.random.default_rng(seed=random_state)
    shuffled_indices = rng.permutation(np.arange(n_samples))
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]
    return train_indices, test_indices
