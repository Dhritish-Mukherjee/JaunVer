"""JaunVer: From-Scratch Machine Learning Pipeline & Interactive Benchmark.

Executes data loading, leakage-free preprocessing, training of Logistic Regression,
Perceptron, and K-Nearest Neighbors, side-by-side evaluation, and interactive
patient biomarker risk assessment.
"""

from pathlib import Path
import sys
from typing import Dict, Any

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import get_dataset_summary, load_raw_data
from src.metrics.classification import (
    classification_report,
    confusion_matrix,
)
from src.models.knn import KNearestNeighbors
from src.models.logistic_regression import LogisticRegression
from src.models.perceptron import Perceptron
from src.preprocessing.pipeline import clean_and_preprocess_data


# Preset patient biomarker profiles for instant testing
PRESETS: Dict[str, Dict[str, Any]] = {
    "1": {
        "title": "Acute Liver Injury (Severely Elevated Bilirubin & Enzymes)",
        "values": {
            "age": 62.0, "gender": 1.0, "total_bilirubin": 10.9, "direct_bilirubin": 5.5,
            "alkaline_phosphotase": 699.0, "alamine_aminotransferase": 64.0,
            "aspartate_aminotransferase": 100.0, "total_proteins": 7.5, "albumin": 3.2,
            "albumin_and_globulin_ratio": 0.74,
        },
    },
    "2": {
        "title": "Healthy Patient Profile (All Markers Within Reference Intervals)",
        "values": {
            "age": 32.0, "gender": 0.0, "total_bilirubin": 0.8, "direct_bilirubin": 0.2,
            "alkaline_phosphotase": 160.0, "alamine_aminotransferase": 18.0,
            "aspartate_aminotransferase": 20.0, "total_proteins": 7.2, "albumin": 3.8,
            "albumin_and_globulin_ratio": 1.15,
        },
    },
    "3": {
        "title": "Borderline / Mildly Elevated Biomarkers",
        "values": {
            "age": 45.0, "gender": 1.0, "total_bilirubin": 1.9, "direct_bilirubin": 0.9,
            "alkaline_phosphotase": 235.0, "alamine_aminotransferase": 45.0,
            "aspartate_aminotransferase": 55.0, "total_proteins": 6.7, "albumin": 3.0,
            "albumin_and_globulin_ratio": 0.80,
        },
    },
}


def format_confusion_matrix_text(cm: np.ndarray) -> str:
    """Format 2x2 confusion matrix into a readable ASCII representation."""
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    return (
        f"                 Predicted Negative (0)    Predicted Positive (1)\n"
        f"  Actual Neg (0)         TN={tn:<6}                 FP={fp:<6}\n"
        f"  Actual Pos (1)         FN={fn:<6}                 TP={tp:<6}"
    )


def print_metrics_table(title: str, report: dict) -> None:
    """Print formatted metrics card."""
    print(f"\n--- {title} ---")
    print(f"Accuracy:    {report['accuracy'] * 100:.2f}%")
    print(f"Precision:   {report['precision'] * 100:.2f}%")
    print(f"Recall:      {report['recall'] * 100:.2f}%")
    print(f"Specificity: {report['specificity'] * 100:.2f}%")
    print(f"F1-Score:    {report['f1_score'] * 100:.2f}%")


def evaluate_patient_sample(
    profile_name: str,
    p: Dict[str, float],
    scaler: Any,
    knn: KNearestNeighbors,
    lr: LogisticRegression,
    perc: Perceptron,
) -> None:
    """Assess and display risk predictions for a patient profile across all models."""
    print(f"\nEvaluating Profile: {profile_name}")
    print("-" * 72)
    gender_str = "Male" if p["gender"] == 1.0 else "Female"
    print(f"  Biomarkers: Age={int(p['age'])}, Gender={gender_str}")
    print(f"  Total Bilirubin: {p['total_bilirubin']} mg/dL | Direct Bilirubin: {p['direct_bilirubin']} mg/dL")
    print(f"  ALP: {p['alkaline_phosphotase']} IU/L | ALT (SGPT): {p['alamine_aminotransferase']} IU/L | AST (SGOT): {p['aspartate_aminotransferase']} IU/L")
    print(f"  Total Proteins: {p['total_proteins']} g/dL | Albumin: {p['albumin']} g/dL | A/G Ratio: {p['albumin_and_globulin_ratio']}")

    vec = np.array([[
        p["age"], p["gender"], p["total_bilirubin"], p["direct_bilirubin"],
        p["alkaline_phosphotase"], p["alamine_aminotransferase"],
        p["aspartate_aminotransferase"], p["total_proteins"], p["albumin"],
        p["albumin_and_globulin_ratio"],
    ]], dtype=np.float64)

    svec = scaler.transform(vec)

    # 1. KNN
    knn_p = float(knn.predict_proba(svec)[0])
    knn_pred = "HIGH RISK (Liver Disease)" if knn_p >= 0.5 else "LOW RISK (Healthy)"
    dists, _ = knn.kneighbors(svec)
    dist_str = ", ".join([f"{d:.2f}" for d in dists[0]])

    # 2. Logistic Regression
    lr_p = float(lr.predict_proba(svec)[0])
    lr_pred = "HIGH RISK (Liver Disease)" if lr_p >= 0.5 else "LOW RISK (Healthy)"

    # 3. Perceptron
    pscore = float(perc.decision_function(svec)[0])
    perc_pred = "HIGH RISK (Liver Disease)" if pscore >= 0.0 else "LOW RISK (Healthy)"

    print("\nModel Risk Assessments:")
    print(f"  * KNN (k=5):             {knn_pred} | P(Disease) = {knn_p * 100:.1f}%")
    print(f"                           5-Nearest Neighbor Distances: [{dist_str}]")
    print(f"  * Logistic Regression:   {lr_pred} | P(Disease) = {lr_p * 100:.1f}%")
    print(f"  * Perceptron:            {perc_pred} | Linear Decision Score = {pscore:+.2f}")
    print("-" * 72)


def prompt_float(prompt_text: str, default: float) -> float:
    """Helper to safely prompt user for float input with robust default fallback."""
    try:
        val = input(f"  {prompt_text} [default: {default}]: ").strip()
        return float(val) if val else default
    except ValueError:
        print(f"    Invalid input. Using default: {default}")
        return default


def main() -> None:
    print("=" * 76)
    print("         JAUNVER: FROM-SCRATCH ML BENCHMARK & INFERENCE RUNNER")
    print("=" * 76)

    # 1. Load Data
    raw_path = Path("data/raw/indian_liver_patient.csv")
    print(f"\n[1] Loading dataset from: {raw_path}")
    raw_df = load_raw_data(raw_path)
    summary = get_dataset_summary(raw_df)
    print(f"    Raw Shape: {summary['total_samples']} rows, {summary['total_columns']} columns")
    print(f"    Duplicates detected: {summary['duplicate_rows']}")
    print(f"    Missing values: {summary['missing_values']}")
    print(f"    Class distribution: {summary['target_counts']} ({summary['target_proportions']})")

    # 2. Preprocessing
    print("\n[2] Executing Leakage-Free Preprocessing...")
    prep = clean_and_preprocess_data(raw_df, test_size=0.2, random_state=42)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    scaler = prep["scaler"]

    print(f"    Dropped {prep['duplicates_dropped']} duplicates -> 570 clean samples.")
    print(f"    Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"    Train Imputation Median (A/G Ratio): {prep['train_median_ag_ratio']:.4f}")
    print(f"    Train target positive ratio: {np.mean(y_train == 1) * 100:.2f}%")
    print(f"    Test target positive ratio:  {np.mean(y_test == 1) * 100:.2f}%")

    # 3. Model 1: Logistic Regression
    print("\n[3] Training Logistic Regression (Batch Gradient Descent)...")
    lr = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
    lr.fit(X_train, y_train)

    initial_loss = lr.loss_history_[0]
    final_loss = lr.loss_history_[-1]
    print(f"    Initial BCE Loss: {initial_loss:.4f} -> Final BCE Loss: {final_loss:.4f}")

    lr_train_preds = lr.predict(X_train)
    lr_test_preds = lr.predict(X_test)

    lr_train_report = classification_report(y_train, lr_train_preds)
    lr_test_report = classification_report(y_test, lr_test_preds)

    print_metrics_table("Logistic Regression (Test Set Evaluation)", lr_test_report)
    cm_lr = confusion_matrix(y_test, lr_test_preds)
    print("\nConfusion Matrix (Logistic Regression Test Set):")
    print(format_confusion_matrix_text(cm_lr))

    # 4. Model 2: Perceptron
    print("\n[4] Training Perceptron Classifier...")
    perc = Perceptron(learning_rate=0.01, n_iterations=100, random_state=42)
    perc.fit(X_train, y_train)

    perc_train_preds = perc.predict(X_train)
    perc_test_preds = perc.predict(X_test)

    perc_train_report = classification_report(y_train, perc_train_preds)
    perc_test_report = classification_report(y_test, perc_test_preds)

    print_metrics_table("Perceptron (Test Set Evaluation)", perc_test_report)
    cm_perc = confusion_matrix(y_test, perc_test_preds)
    print("\nConfusion Matrix (Perceptron Test Set):")
    print(format_confusion_matrix_text(cm_perc))

    # 5. Model 3: K-Nearest Neighbors (k=5)
    print("\n[5] Evaluating K-Nearest Neighbors (k=5)...")
    knn = KNearestNeighbors(n_neighbors=5)
    knn.fit(X_train, y_train)

    knn_test_preds = knn.predict(X_test)
    knn_test_report = classification_report(y_test, knn_test_preds)

    print_metrics_table("KNN (Test Set Evaluation)", knn_test_report)
    cm_knn = confusion_matrix(y_test, knn_test_preds)
    print("\nConfusion Matrix (KNN Test Set):")
    print(format_confusion_matrix_text(cm_knn))

    # 6. Side-by-Side Comparison
    print("\n" + "=" * 82)
    print(f"{'Metric':<18} | {'Logistic Regression':<20} | {'Perceptron':<16} | {'KNN (k=5)':<16}")
    print("-" * 82)
    for metric in ["accuracy", "precision", "recall", "specificity", "f1_score"]:
        lr_val = lr_test_report[metric] * 100
        perc_val = perc_test_report[metric] * 100
        knn_val = knn_test_report[metric] * 100
        print(f"{metric.replace('_', ' ').capitalize():<18} | {lr_val:>18.2f}% | {perc_val:>14.2f}% | {knn_val:>14.2f}%")
    print("=" * 82)

    # 7. Save Convergence Figure
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    loss_plot_path = fig_dir / "logistic_regression_loss.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(lr.loss_history_) + 1), lr.loss_history_, color="#1f77b4", lw=2)
    plt.title("Logistic Regression: Batch Gradient Descent Convergence (ILPD)")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"\n[6] Saved gradient descent loss curve to: {loss_plot_path}")

    # 8. Scientific Findings
    print("\n[7] Scientific Findings & Clinical Interpretability:")
    print("    - High Recall (~85-97%): Catches the vast majority of true liver disease patients.")
    print("    - Class Imbalance Impact: ~71% diseased vs 29% healthy makes specificity harder.")
    print("    - Model Behaviors:")
    print("      * Logistic Regression produces smooth, calibrated posterior probabilities.")
    print("      * KNN (k=5) learns non-linear local patient clusters via Euclidean distance.")
    print("      * Perceptron requires linear separability and oscillates on overlapping markers.")

    # 9. Demonstration Patient Risk Assessment
    print("\n" + "=" * 76)
    print("         [8] LIVE CLINICAL PATIENT INFERENCE DEMO")
    print("=" * 76)
    evaluate_patient_sample(
        "Acute Liver Injury (Pre-configured Baseline)",
        PRESETS["1"]["values"],
        scaler, knn, lr, perc
    )

    # 10. Interactive Testing Mode (if running in interactive terminal)
    if sys.stdin.isatty():
        while True:
            print("\nInteractive Terminal Testing Menu:")
            print("  [1] Test Preset: Acute Liver Injury (High Risk)")
            print("  [2] Test Preset: Healthy Profile (Low Risk)")
            print("  [3] Test Preset: Borderline Profile")
            print("  [4] Enter Custom Biomarker Values Manually")
            print("  [Press Enter to Exit]")

            choice = input("\nSelect an option [1-4 or Enter]: ").strip()
            if not choice:
                print("\nExiting interactive assessment.")
                break

            if choice in PRESETS:
                preset = PRESETS[choice]
                evaluate_patient_sample(
                    preset["title"],
                    preset["values"],
                    scaler, knn, lr, perc
                )
            elif choice == "4":
                print("\nEnter patient biomarker values:")
                c_age = prompt_float("Age", 45.0)
                c_gen = prompt_float("Gender (1=Male, 0=Female)", 1.0)
                c_tb = prompt_float("Total Bilirubin (mg/dL)", 1.0)
                c_db = prompt_float("Direct Bilirubin (mg/dL)", 0.3)
                c_alp = prompt_float("Alkaline Phosphatase (IU/L)", 180.0)
                c_alt = prompt_float("ALT / SGPT (IU/L)", 25.0)
                c_ast = prompt_float("AST / SGOT (IU/L)", 30.0)
                c_tp = prompt_float("Total Proteins (g/dL)", 7.0)
                c_ab = prompt_float("Albumin (g/dL)", 3.5)
                c_ag = prompt_float("A/G Ratio", 1.0)

                custom_vals = {
                    "age": c_age, "gender": c_gen, "total_bilirubin": c_tb,
                    "direct_bilirubin": c_db, "alkaline_phosphotase": c_alp,
                    "alamine_aminotransferase": c_alt, "aspartate_aminotransferase": c_ast,
                    "total_proteins": c_tp, "albumin": c_ab,
                    "albumin_and_globulin_ratio": c_ag,
                }
                evaluate_patient_sample(
                    "Custom Clinician Input",
                    custom_vals,
                    scaler, knn, lr, perc
                )
            else:
                print("Invalid selection. Please choose 1, 2, 3, 4, or press Enter to exit.")

    print("\nJaunVer benchmark execution finished.\n")


if __name__ == "__main__":
    main()
