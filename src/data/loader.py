"""Dataset loading and schema definition for the Indian Liver Patient Dataset (ILPD).

This module provides reproducible loading of raw tabular data with explicit
feature definitions and schema validation.
"""

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

# Canonical feature names for ILPD (UCI repository standard)
FEATURE_COLUMNS: List[str] = [
    "age",
    "gender",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "total_proteins",
    "albumin",
    "albumin_and_globulin_ratio",
]

TARGET_COLUMN: str = "dataset"

ALL_COLUMNS: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_raw_data(
    filepath: Union[str, Path] = "data/raw/indian_liver_patient.csv"
) -> pd.DataFrame:
    """Load the raw ILPD CSV file and assign standard column headers.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with 11 named columns and original raw records.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the specified path.
    ValueError
        If the loaded data does not have exactly 11 columns.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)

    if df.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns, but found {df.shape[1]}."
        )

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute structural metrics and class distribution of the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded ILPD dataframe.

    Returns
    -------
    Dict[str, object]
        Summary containing shape, missing values, duplicates, and target counts.
    """
    total_samples, total_cols = df.shape
    missing_counts = df.isna().sum().to_dict()
    missing_only = {col: count for col, count in missing_counts.items() if count > 0}
    duplicate_rows = int(df.duplicated().sum())

    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    total = len(df)
    target_proportions = {
        cls: round(count / total, 4) for cls, count in target_counts.items()
    }

    return {
        "total_samples": total_samples,
        "total_columns": total_cols,
        "missing_values": missing_only,
        "duplicate_rows": duplicate_rows,
        "target_counts": target_counts,
        "target_proportions": target_proportions,
    }
