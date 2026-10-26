"""Export trained from-scratch models and preprocessing statistics for deployment.

Saves:
1. 'artifacts/model_bundle.pkl': Full serialized Python objects for instant serving.
2. 'artifacts/model_parameters.json': Human-readable weights, biases, and scaling parameters.
"""

import json
from pathlib import Path
import pickle
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_raw_data
from src.models.knn import KNearestNeighbors
from src.models.logistic_regression import LogisticRegression
from src.models.perceptron import Perceptron
from src.preprocessing.pipeline import clean_and_preprocess_data


# Standard adult clinical reference ranges for educational context
CLINICAL_REFERENCE_RANGES = {
    "total_bilirubin": {"min": 0.1, "max": 1.2, "unit": "mg/dL"},
    "direct_bilirubin": {"min": 0.0, "max": 0.3, "unit": "mg/dL"},
    "alkaline_phosphotase": {"min": 44.0, "max": 147.0, "unit": "IU/L"},
    "alamine_aminotransferase": {"min": 7.0, "max": 56.0, "unit": "IU/L"},
    "aspartate_aminotransferase": {"min": 10.0, "max": 40.0, "unit": "IU/L"},
    "total_proteins": {"min": 6.0, "max": 8.3, "unit": "g/dL"},
    "albumin": {"min": 3.4, "max": 5.4, "unit": "g/dL"},
    "albumin_and_globulin_ratio": {"min": 0.8, "max": 2.0, "unit": "ratio"},
}


def export_models() -> Path:
    """Train models on training split and serialize bundle to disk."""
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Preprocess Data
    raw_df = load_raw_data("data/raw/indian_liver_patient.csv")
    prep = clean_and_preprocess_data(raw_df, test_size=0.2, random_state=42)

    X_train, y_train = prep["X_train"], prep["y_train"]

    # 2. Train Models
    print("Training Logistic Regression...")
    lr = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
    lr.fit(X_train, y_train)

    print("Training Perceptron...")
    perc = Perceptron(learning_rate=0.01, n_iterations=100, random_state=42)
    perc.fit(X_train, y_train)

    print("Fitting K-Nearest Neighbors (k=5)...")
    knn = KNearestNeighbors(n_neighbors=5)
    knn.fit(X_train, y_train)

    # 3. Create Artifact Bundle
    bundle = {
        "scaler": prep["scaler"],
        "train_median_ag_ratio": prep["train_median_ag_ratio"],
        "feature_names": prep["feature_names"],
        "logistic_regression": lr,
        "perceptron": perc,
        "knn": knn,
        "clinical_reference_ranges": CLINICAL_REFERENCE_RANGES,
    }

    bundle_path = artifacts_dir / "model_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Serialized model bundle saved to: {bundle_path}")

    # 4. Also export human-readable parameters in JSON for viva transparency
    json_path = artifacts_dir / "model_parameters.json"
    params_export = {
        "feature_names": prep["feature_names"],
        "imputation": {
            "albumin_and_globulin_ratio_median": prep["train_median_ag_ratio"],
        },
        "scaler": {
            "means": prep["scaler"].mean_.tolist(),
            "stds": prep["scaler"].std_.tolist(),
        },
        "logistic_regression": {
            "weights": lr.weights_.tolist(),
            "bias": float(lr.bias_),
            "learning_rate": lr.learning_rate,
            "n_iterations": lr.n_iterations,
        },
        "perceptron": {
            "weights": perc.weights_.tolist(),
            "bias": float(perc.bias_),
            "learning_rate": perc.learning_rate,
            "n_iterations": perc.n_iterations,
        },
        "knn": {
            "n_neighbors": knn.n_neighbors,
            "metric": "euclidean",
            "n_training_samples": len(X_train),
        },
    }

    with open(json_path, "w") as f:
        json.dump(params_export, f, indent=2)
    print(f"Human-readable parameters saved to: {json_path}")

    return bundle_path


if __name__ == "__main__":
    export_models()
