"""Unit tests for dataset loading and schema verification."""

import pytest
from src.data.loader import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_dataset_summary,
    load_raw_data,
)


def test_load_raw_data_shape_and_columns():
    """Verify that dataset loads with shape (583, 11) and correct column names."""
    df = load_raw_data("data/raw/indian_liver_patient.csv")
    assert df.shape == (583, 11)
    assert list(df.columns) == ALL_COLUMNS
    assert len(FEATURE_COLUMNS) == 10
    assert TARGET_COLUMN in df.columns


def test_target_values():
    """Verify that raw target labels are strictly 1 (diseased) and 2 (healthy)."""
    df = load_raw_data("data/raw/indian_liver_patient.csv")
    unique_targets = set(df[TARGET_COLUMN].unique())
    assert unique_targets == {1, 2}
    counts = df[TARGET_COLUMN].value_counts().to_dict()
    assert counts[1] == 416
    assert counts[2] == 167


def test_missing_values_and_duplicates_summary():
    """Verify missing values and duplicate rows match UCI specifications."""
    df = load_raw_data("data/raw/indian_liver_patient.csv")
    summary = get_dataset_summary(df)

    assert summary["total_samples"] == 583
    assert summary["total_columns"] == 11
    # Exactly 4 missing values in albumin_and_globulin_ratio
    assert summary["missing_values"] == {"albumin_and_globulin_ratio": 4}
    # Exactly 13 duplicate rows
    assert summary["duplicate_rows"] == 13


def test_load_raw_data_missing_file_raises():
    """Verify that attempting to load a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/non_existent_file.csv")
