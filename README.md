# JaunVer: Liver Disease Classification System

An educational Machine Learning project built from scratch using NumPy and Python to classify liver disease using the Indian Liver Patient Dataset (ILPD).

> **Academic & Scientific Disclaimer:**
> This repository is an educational research and coursework project. It is **not** a clinical diagnostic medical device. Model outputs represent statistical associations on retrospective tabular data and must not be used for patient diagnosis or treatment decisions.

---

## Project Structure

```
JaunVer/
├── data/
│   └── raw/
│       └── indian_liver_patient.csv    # Original raw dataset (583 records, 11 features)
├── assets/
│   ├── jaunver.png                     # Official JaunVer logo
│   └── jaunver_icon.svg                # Browser tab icon
├── src/
│   ├── data/
│   │   └── loader.py                   # Schema definition, validation, and metadata summary
│   ├── preprocessing/
│   │   ├── scaler.py                   # From-scratch StandardScaler (Z-score normalization)
│   │   └── pipeline.py                 # Leakage-free train/test split and train-median imputation
│   ├── models/
│   │   ├── logistic_regression.py     # From-scratch Logistic Regression (Batch Gradient Descent)
│   │   ├── perceptron.py               # From-scratch Rosenblatt Perceptron classifier
│   │   └── knn.py                      # From-scratch K-Nearest Neighbors classifier (Euclidean distance)
│   └── metrics/
│       └── classification.py           # Confusion matrix, Accuracy, Precision, Recall, F1
├── artifacts/
│   ├── model_bundle.pkl                # Serialized preprocessing and model artifacts
│   └── model_parameters.json           # Human-readable weights, biases, and scaling parameters
├── web/
│   └── app.py                          # Streamlit web application
├── scripts/
│   ├── run_phase1.py                   # End-to-end command-line evaluation pipeline
│   └── export_models.py                # Model training and artifact serialization script
├── tests/
│   ├── test_data.py                    # Schema and data loading tests
│   ├── test_preprocessing.py           # Scaling math and leakage tests
│   ├── test_models.py                  # Analytical gradient and loss convergence tests
│   ├── test_metrics.py                 # Hand-calculated metric tests
│   └── test_web.py                     # Web bundle loading and inference tests
├── reports/
│   └── figures/
│       └── logistic_regression_loss.png # Optimization convergence curve
├── requirements.txt                    # Project dependencies
├── PLAN.md                             # 3-Phase milestone roadmap
└── AGENTS.md                           # Project instructions and academic integrity rules
```

---

## Quickstart

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Run Test Suite (19 Unit Tests)
```bash
pytest -v
```

### 3. Run Command-Line Pipeline
```bash
python scripts/run_phase1.py
```

### 4. Launch Interactive Web Dashboard
```bash
streamlit run web/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## Deploying to Your Paid VPS (Ubuntu/Debian)

Run these commands on your VPS terminal:

```bash
# 1. Clone repository
git clone <your-repo-url>
cd JaunVer

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Export model artifacts
python scripts/export_models.py

# 4. Launch Streamlit 24/7 in background
nohup streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```
The app will be accessible to your teacher at:
`http://<YOUR_VPS_PUBLIC_IP>:8501`
*(Ensure port 8501 is allowed in your VPS firewall / UFW: `sudo ufw allow 8501/tcp`).*
