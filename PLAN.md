# JaunVer: Step-by-Step Project Plan

**Goal:** Build a complete, from-scratch Machine Learning classification system for liver disease prediction without external ML libraries (no `scikit-learn`), fully prepared for oral viva defense within a 4-day deadline.

---

## Architecture & Guiding Philosophy

1. **Safety First (Fail-Safe Submissions):**
   - **Phase 1 (Basic)** produces a 100% complete, working, submittable assignment. If unexpected emergencies happen, you can submit Phase 1 and get full marks for core requirements.
   - **Phase 2 (Intermediate)** adds multi-model benchmarking and cross-validation to demonstrate breadth.
   - **Phase 3 (Advanced)** adds backpropagation/MLP and comprehensive ROC analysis to aim for top tier / extra credit.

2. **Viva Defense Ready:**
   - Every algorithm is implemented manually using NumPy/Python.
   - For every component, we explain the mathematics, hand-work an example, analyze failure modes, and run quick viva quizzes before moving forward.

3. **No External ML Frameworks:**
   - Permitted: NumPy (vector math), Pandas (tabular I/O, if allowed), Matplotlib (plots).
   - Forbidden: `sklearn`, `scipy.stats` model fits, `torch`, `tensorflow`.

---

## Phase 1: Minimum Viable Assignment (Absolute Basic — Target: Day 1–2)

> **Objective:** Deliver a working, end-to-end pipeline (data loading, preprocessing, Logistic Regression + Perceptron, metrics, and unit tests) that can be submitted immediately.

### Step 1.1: Environment, Dataset & Project Setup
- Establish clean directory layout (`src/data`, `src/models`, `src/metrics`, `src/utils`, `tests/`).
- Place and inspect the dataset (typically Indian Liver Patient Dataset / ILPD).
- Document feature schemas, missing values, and binary label mapping (e.g., patient vs. non-patient $\to 1$ and $0$).
- **Viva Concept:** Feature types (continuous vs. discrete), clinical meaning of liver enzymes (ALT, AST, Bilirubin).

### Step 1.2: From-Scratch Data Preprocessing & Splitting
- Implement manual random / stratified train-test split with fixed random seed (`train_test_split`).
- Handle missing values (e.g., median imputation for `Albumin_and_Globulin_Ratio`).
- One-hot / binary encoding for categorical features (`Gender`).
- Implement Feature Standardization (Z-score: $z = \frac{x - \mu}{\sigma}$) fitted **strictly on train set** to prevent data leakage.
- **Viva Concept:** What is data leakage? Why must $\mu$ and $\sigma$ come only from training data?

### Step 1.3: Logistic Regression with Batch Gradient Descent (From Scratch)
- Sigmoid activation function: $\sigma(z) = \frac{1}{1 + e^{-z}}$ (with numerical clipping to prevent overflow).
- Binary Cross-Entropy Loss:
  $$\mathcal{L}(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
- Analytic gradient derivations:
  $$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$
- Parameter updates via Gradient Descent.
- **Viva Concept:** Convexity of BCE loss vs. non-convexity of MSE for classification; learning rate tuning; gradient derivation.

### Step 1.4: Perceptron Classifier (From Scratch)
- Linear decision boundary: $z = w^T x + b$.
- Step activation function: $\hat{y} = 1$ if $z \ge 0$ else $0$.
- Perceptron update rule for misclassified points:
  $$w \leftarrow w + \eta (y - \hat{y}) x, \quad b \leftarrow b + \eta (y - \hat{y})$$
- **Viva Concept:** Perceptron convergence theorem, linear separability limitation, difference between Perceptron and Logistic Regression.

### Step 1.5: Core Evaluation Metrics (From Scratch)
- Binary Confusion Matrix: True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN).
- Accuracy, Precision, Recall (Sensitivity), Specificity, and F1-Score formulas:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Viva Concept:** Medical cost of False Negatives (missing a diseased patient) vs False Positives; why accuracy is misleading under class imbalance.

### Step 1.6: Phase 1 Integration & Unit Testing
- Test suite verifying mathematical functions (Sigmoid, BCE loss, gradients, standardization, metrics) using hand-worked assertions.
- Main runner script executing Phase 1 end-to-end and printing formatted results.

---

## Phase 2: Intermediate Tier (Model Diversity & Robustness — Target: Day 2–3)

> **Objective:** Add non-parametric and probabilistic models, manual K-Fold Cross Validation, and class imbalance mitigation.

### Step 2.1: K-Nearest Neighbors (KNN) (From Scratch) - [COMPLETED]
- Vectorized Euclidean distance computation between test query and training instances.
- Efficient sorting / top-$k$ selection.
- Majority voting mechanism and tie-breaking strategy.
- Tuning parameter $k$ to balance bias vs. variance.
- **Viva Concept:** Curse of dimensionality; why feature scaling is critical for KNN; computational complexity at test time $\mathcal{O}(N \cdot D)$.

### Step 2.2: Gaussian Naive Bayes (From Scratch)
- Prior class probabilities: $P(Y = c)$.
- Maximum Likelihood Estimation of mean $\mu_{c,j}$ and variance $\sigma_{c,j}^2$ per feature $j$ per class $c$.
- Gaussian probability density function (PDF):
  $$P(X_j = x_j \mid Y = c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}\right)$$
- Log-likelihood formulation to prevent floating-point underflow.
- **Viva Concept:** The "naive" conditional independence assumption; handling zero variance / epsilon smoothing.

### Step 2.3: Manual Stratified K-Fold Cross-Validation
- Partitioning training data into $K$ equal-sized folds preserving class ratios.
- Training on $K-1$ folds and evaluating on the held-out fold iteratively.
- Computing mean and standard deviation of validation metrics across folds.
- **Viva Concept:** Why cross-validation gives a more reliable estimate than a single train-test split; variance of estimators.

### Step 2.4: Addressing Class Imbalance
- Liver datasets are typically unbalanced (~70% diseased, ~30% healthy).
- Implement class-weighted loss in Logistic Regression:
  $$w_c = \frac{N}{2 \cdot N_c}$$
- Decision threshold tuning (moving classification boundary from $0.5$ to optimize Recall/F1).
- **Viva Concept:** Cost-sensitive learning; threshold moving vs. resampling.

---

## Phase 3: Advanced Tier (Deep Learning & Comprehensive Benchmarking — Target: Day 3–4)

> **Objective:** Multilayer Perceptron with backpropagation, full ROC-AUC evaluation, and comprehensive viva preparation report.

### Step 3.1: Multilayer Perceptron (MLP) with Backpropagation (From Scratch)
- Architecture: Input Layer $\to$ Hidden Layer (ReLU / Leaky ReLU) $\to$ Output Layer (Sigmoid).
- Forward Propagation:
  $$Z^{[1]} = X W^{[1]} + b^{[1]}, \quad A^{[1]} = \text{ReLU}(Z^{[1]})$$
  $$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}, \quad \hat{Y} = \sigma(Z^{[2]})$$
- Backpropagation via Chain Rule:
  $$\delta^{[2]} = \hat{Y} - Y$$
  $$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{1}{m} (A^{[1]})^T \delta^{[2]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m} \sum \delta^{[2]}$$
  $$\delta^{[1]} = (\delta^{[2]} (W^{[2]})^T) \odot g'(Z^{[1]})$$
  $$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{1}{m} X^T \delta^{[1]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m} \sum \delta^{[1]}$$
- Mini-batch gradient descent and weight initialization (He / Xavier).
- **Viva Concept:** Vanishing/exploding gradients; role of activation functions; step-by-step chain rule derivation.

### Step 3.2: ROC Curve & AUC Score (From Scratch)
- Implement sorting of prediction probabilities.
- Iterative computation of True Positive Rate (TPR) and False Positive Rate (FPR) across all unique thresholds.
- Trapezoidal rule integration to compute Area Under Curve (AUC).
- **Viva Concept:** Difference between Precision-Recall curve and ROC curve; threshold independence of AUC.

### Step 3.3: Master Benchmark & Visualizations
- Generate a consolidated performance table comparing all 5 algorithms:
  1. Logistic Regression (Baseline & Weighted)
  2. Perceptron
  3. K-Nearest Neighbors
  4. Gaussian Naive Bayes
  5. Multilayer Perceptron
- Generate publication-quality comparison charts (ROC curves, metric bar charts) using Matplotlib.

### Step 3.4: Viva Defense Dossier & Scientific Disclaimer
- Document clinical caveats: screening aid vs diagnostic standard.
- Formulate a Q&A viva cheat sheet covering likely examiner traps (overfitting, feature collinearity like ALT vs AST, metric selection, convergence rates).

---

## Workflow Rules for Every Single Step

For every sub-step we execute together:
1. **Explain the Theory & Math:** Exact equations and hand-worked numerical walkthrough.
2. **Review Design & Plan:** File location, function signatures, input/output shapes.
3. **Wait for Student Approval:** No code written until you say go.
4. **Implement Clean Code:** Clear comments, no magic numbers, clean typing.
5. **Run Tests & Verify:** Verify against hand-calculated values.
6. **Viva Checkpoint:** 2–3 questions to ensure you master the concepts before we proceed.
