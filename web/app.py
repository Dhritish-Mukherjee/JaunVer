"""JaunVer — Interactive ML Pipeline Dashboard.

Five-stage walkthrough: data inspection, preprocessing, training,
evaluation, and live inference for liver disease classification.
"""

from pathlib import Path
import base64
import pickle
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_dataset_summary,
    load_raw_data,
)
from src.metrics.classification import classification_report, confusion_matrix
from src.models.knn import KNearestNeighbors
from src.models.logistic_regression import LogisticRegression
from src.models.perceptron import Perceptron
from src.preprocessing.pipeline import clean_and_preprocess_data

# ─── Page Setup ────────────────────────────────────────────
ICON_PATH = REPO_ROOT / "assets" / "jaunver_icon.svg"
WEB_LOGO = REPO_ROOT / "assets" / "jaunver_web.png"
LOGO_PATH = WEB_LOGO if WEB_LOGO.is_file() else (REPO_ROOT / "assets" / "jaunver.png")

st.set_page_config(
    page_title="JaunVer",
    page_icon=str(ICON_PATH) if ICON_PATH.is_file() else "H",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def get_image_base64(path_str: str) -> str:
    """Read an image file and return its base64 data string to preserve native resolution."""
    p = Path(path_str)
    if not p.is_file():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ─── Plotly Theme ──────────────────────────────────────────
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
TEXT_COLOR = "#B0B8C4"
ACCENT_BLUE = "#4A90D9"
ACCENT_ORANGE = "#E6863A"
ACCENT_RED = "#D94A4A"
ACCENT_GREEN = "#3DAF6B"
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def plotly_layout(**overrides):
    """Base plotly layout that blends with streamlit dark theme."""
    base = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=13),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        hoverlabel=dict(bgcolor="#1A1F2B", font_color="#E8EAED", font_size=12),
    )
    base.update(overrides)
    return go.Layout(**base)


# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* clean metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { font-weight: 600; }

    /* tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 500;
        font-size: 0.9rem;
        color: #8892A0;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #E8EAED !important;
        border-bottom: 2px solid #4A90D9 !important;
    }

    /* section dividers */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* dataframe tweaks */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* expander */
    .streamlit-expanderHeader { font-weight: 500; font-size: 0.9rem; }

    /* alert boxes */
    .stAlert > div { border-radius: 8px; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading (cached) ────────────────────────────────
@st.cache_data
def get_raw_data():
    return load_raw_data(REPO_ROOT / "data" / "raw" / "indian_liver_patient.csv")


@st.cache_data
def get_preprocessed():
    return clean_and_preprocess_data(get_raw_data(), test_size=0.2, random_state=42)


@st.cache_resource
def load_bundle():
    bundle_path = REPO_ROOT / "artifacts" / "model_bundle.pkl"
    if not bundle_path.is_file():
        from scripts.export_models import export_models
        export_models()

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    if "knn" not in bundle:
        from scripts.export_models import export_models
        export_models()
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

    return bundle


raw_df = get_raw_data()
prep = get_preprocessed()
bundle = load_bundle()
lr_model = bundle["logistic_regression"]
perc_model = bundle["perceptron"]
knn_model = bundle["knn"]
X_train, X_test = prep["X_train"], prep["X_test"]
y_train, y_test = prep["y_train"], prep["y_test"]
feature_names = prep["feature_names"]


# ─── Sidebar ──────────────────────────────────────────────
logo_b64 = get_image_base64(str(LOGO_PATH)) if LOGO_PATH.is_file() else ""
if logo_b64:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <img src="data:image/png;base64,{logo_b64}" 
                 alt="JaunVer Logo" 
                 style="height: 48px; width: auto; max-width: 52px; object-fit: contain; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));" />
            <div>
                <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700; line-height: 1.15; color: #FFFFFF;">JaunVer</h2>
                <span style="font-size: 0.78rem; color: #8892A0;">Liver Disease ML</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.title("JaunVer")
    st.sidebar.caption("Liver Disease Classification System")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Coursework:** B.Tech Project  \n"
    "**Dataset:** Indian Liver Patient Dataset (UCI)  \n"
    "**Models:** Logistic Regression, Perceptron, K-NN (k=5)  \n"
    "**Engine:** 100% From-Scratch NumPy"
)

# ─── Header ───────────────────────────────────────────────
if logo_b64:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 18px; margin-bottom: 1.25rem;">
            <img src="data:image/png;base64,{logo_b64}" 
                 alt="JaunVer Logo" 
                 style="height: 68px; width: auto; max-width: 80px; object-fit: contain; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.35));" />
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1.15; color: #FFFFFF;">JaunVer</h1>
                <p style="color: #8892A0; font-size: 0.95rem; margin: 4px 0 0 0;">
                    End-to-end liver disease classification pipeline — implemented from scratch in NumPy
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 4px; font-weight: 700; letter-spacing: -0.5px;">JaunVer</h1>
        <p style="color: #8892A0; font-size: 1rem; margin-top: 0;">
            End-to-end liver disease classification pipeline — implemented from scratch in NumPy
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset", "Preprocessing", "Training", "Evaluation", "Predict",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ══════════════════════════════════════════════════════════
with tab1:
    summary = get_dataset_summary(raw_df)

    st.markdown("#### Dataset Summary")
    st.caption("Indian Liver Patient Dataset (ILPD) — UCI Machine Learning Repository")
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", summary["total_samples"])
    c2.metric("Features", summary["total_columns"] - 1)
    c3.metric("Duplicates", summary["duplicate_rows"])
    c4.metric("Missing", sum(summary["missing_values"].values()))

    st.markdown("")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("##### Class Distribution")
        counts = [summary["target_counts"][1], summary["target_counts"][2]]
        labels = ["Liver Disease", "Healthy"]

        fig_dist = go.Figure(go.Bar(
            x=labels, y=counts,
            marker=dict(
                color=[ACCENT_RED, ACCENT_GREEN],
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{c}  ({c/sum(counts)*100:.1f}%)" for c in counts],
            textposition="outside",
            textfont=dict(size=13, color=TEXT_COLOR),
        ))
        fig_dist.update_layout(
            plotly_layout(
                height=320,
                yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, title=None),
                xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None),
                showlegend=False,
            )
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.info(
            "**Class imbalance:** 71.4% of records are liver disease patients. "
            "Accuracy alone is insufficient — Recall and F1-Score are used as primary metrics."
        )

    with col_right:
        st.markdown("##### Feature Dictionary")
        ref = pd.DataFrame([
            {"Feature": "Age", "Type": "Demographic", "Description": "Patient age in years"},
            {"Feature": "Gender", "Type": "Categorical", "Description": "Male / Female"},
            {"Feature": "Total Bilirubin", "Type": "LFT", "Description": "Elevated in jaundice"},
            {"Feature": "Direct Bilirubin", "Type": "LFT", "Description": "Elevated in biliary obstruction"},
            {"Feature": "ALP", "Type": "Enzyme", "Description": "Bile duct marker"},
            {"Feature": "ALT (SGPT)", "Type": "Enzyme", "Description": "Liver-specific injury marker"},
            {"Feature": "AST (SGOT)", "Type": "Enzyme", "Description": "Hepatocellular damage marker"},
            {"Feature": "Total Proteins", "Type": "Protein", "Description": "Blood protein concentration"},
            {"Feature": "Albumin", "Type": "Protein", "Description": "Decreased in cirrhosis"},
            {"Feature": "A/G Ratio", "Type": "Protein", "Description": "Decreases in chronic liver disease"},
        ])
        st.dataframe(ref, use_container_width=True, hide_index=True, height=390)

    st.markdown("---")
    st.markdown("##### Browse Records")
    n = st.slider("Rows", 5, 100, 10, key="raw_n")
    st.dataframe(raw_df.head(n), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Preprocessing Pipeline")
    st.caption("All statistics are computed exclusively from the training split to prevent data leakage.")

    st.markdown("")

    # Step 1 — Dedup
    st.markdown("##### 1. Deduplication")
    d1, d2 = st.columns(2)
    d1.metric("Before", "583 rows")
    d2.metric("After", "570 rows", delta="-13 duplicates")
    st.markdown(
        "Identical records removed to prevent train-test contamination."
    )

    st.markdown("---")

    # Step 2 — Encoding
    st.markdown("##### 2. Encoding")
    e1, e2 = st.columns(2)
    with e1:
        st.code("Target:  1 (Patient) → 1    |    2 (Healthy) → 0", language="text")
    with e2:
        st.code("Gender:  Male → 1           |    Female → 0", language="text")

    st.markdown("---")

    # Step 3 — Split
    st.markdown("##### 3. Train / Test Split")
    s1, s2, s3 = st.columns(3)
    s1.metric("Train", f"{len(X_train)} (80%)")
    s2.metric("Test", f"{len(X_test)} (20%)")
    s3.metric("Seed", "42")
    st.markdown("Split performed **before** imputation and scaling.")

    st.markdown("---")

    # Step 4 — Imputation
    st.markdown("##### 4. Missing Value Imputation")
    st.markdown(
        f"`albumin_and_globulin_ratio` has 4 missing values, filled with the "
        f"**training median = {prep['train_median_ag_ratio']:.2f}**."
    )

    st.markdown("---")

    # Step 5 — Scaling
    st.markdown("##### 5. Feature Standardization")
    st.latex(r"z_{i,j} = \frac{x_{i,j} - \mu_j^{\,\text{train}}}{\sigma_j^{\,\text{train}}}")

    raw_clean = raw_df.drop_duplicates().copy()
    raw_clean[TARGET_COLUMN] = (raw_clean[TARGET_COLUMN] == 1).astype(int)
    raw_clean["gender"] = (raw_clean["gender"].str.strip().str.lower() == "male").astype(int)
    raw_stats = raw_clean[feature_names].describe().loc[["mean", "std", "min", "max"]].T.round(2)
    scaled_stats = pd.DataFrame(X_train, columns=feature_names).describe().loc[["mean", "std", "min", "max"]].T.round(4)

    sc1, sc2 = st.columns(2, gap="large")
    with sc1:
        st.markdown("**Raw features**")
        st.dataframe(raw_stats, use_container_width=True)
    with sc2:
        st.markdown("**After standardization**")
        st.dataframe(scaled_stats, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — TRAINING
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Model Training")
    st.markdown("")

    lr_tab, perc_tab, knn_tab = st.tabs([
        "Logistic Regression",
        "Perceptron",
        "K-Nearest Neighbors (KNN)",
    ])

    with lr_tab:
        math_l, math_r = st.columns(2)
        with math_l:
            st.markdown("**Forward pass**")
            st.latex(r"\hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}")
            st.markdown("**Loss (Binary Cross-Entropy)**")
            st.latex(
                r"\mathcal{L} = -\frac{1}{m} \sum "
                r"\left[ y \log \hat{y} + (1-y) \log(1-\hat{y}) \right]"
            )
        with math_r:
            st.markdown("**Gradients**")
            st.latex(r"\nabla_w = \frac{1}{m} X^T (\hat{y} - y)")
            st.latex(r"\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum (\hat{y} - y)")
            st.markdown("**Update**")
            st.latex(r"w \leftarrow w - \alpha \nabla_w, \quad b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}")

        st.markdown("---")

        # Loss curve
        epochs = list(range(1, len(lr_model.loss_history_) + 1))
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=lr_model.loss_history_,
            mode="lines",
            line=dict(color=ACCENT_BLUE, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(74,144,217,0.08)",
            hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
        ))
        fig_loss.update_layout(plotly_layout(
            height=360,
            xaxis_title="Epoch",
            yaxis_title="BCE Loss",
            title=dict(text="Convergence", font=dict(size=15)),
        ))
        st.plotly_chart(fig_loss, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Initial Loss", f"{lr_model.loss_history_[0]:.4f}")
        m2.metric("Final Loss", f"{lr_model.loss_history_[-1]:.4f}")
        reduction_pct = (lr_model.loss_history_[0] - lr_model.loss_history_[-1]) / lr_model.loss_history_[0] * 100
        m3.metric("Reduction", f"{reduction_pct:.1f}%")

        st.markdown("---")

        # Weights
        w_df = pd.DataFrame({
            "feature": feature_names,
            "weight": lr_model.weights_,
        }).sort_values("weight")

        colors = [ACCENT_RED if w > 0 else ACCENT_BLUE for w in w_df["weight"]]

        fig_w = go.Figure(go.Bar(
            y=w_df["feature"], x=w_df["weight"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0), cornerradius=4),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_w.update_layout(plotly_layout(
            height=380,
            xaxis_title="Weight",
            title=dict(text="Learned Feature Weights", font=dict(size=15)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        ))
        st.plotly_chart(fig_w, use_container_width=True)

        st.markdown(f"Bias: `{lr_model.bias_:.4f}`")
        st.markdown(
            "Positive weights (red) push the prediction toward liver disease. "
            "Direct Bilirubin and ALT carry the highest weight, consistent with "
            "their role as primary hepatocellular injury markers."
        )

    with perc_tab:
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Decision rule**")
            st.latex(r"\hat{y} = \begin{cases} 1 & w^T x + b \geq 0 \\ 0 & \text{otherwise} \end{cases}")
        with pc2:
            st.markdown("**Update rule (on error)**")
            st.latex(r"w \leftarrow w + \eta(y - \hat{y})x")
            st.latex(r"b \leftarrow b + \eta(y - \hat{y})")

        st.markdown("---")

        ep = list(range(1, len(perc_model.errors_history_) + 1))
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=ep, y=perc_model.errors_history_,
            mode="lines+markers",
            line=dict(color=ACCENT_ORANGE, width=2),
            marker=dict(size=4, color=ACCENT_ORANGE),
            fill="tozeroy",
            fillcolor="rgba(230,134,58,0.08)",
            hovertemplate="Epoch %{x}<br>Errors: %{y}<extra></extra>",
        ))
        fig_err.update_layout(plotly_layout(
            height=360,
            xaxis_title="Epoch",
            yaxis_title="Misclassified Samples",
            title=dict(text="Error Convergence", font=dict(size=15)),
        ))
        st.plotly_chart(fig_err, use_container_width=True)

        pe1, pe2 = st.columns(2)
        pe1.metric("Epoch 1 Errors", perc_model.errors_history_[0])
        pe2.metric("Final Epoch", perc_model.errors_history_[-1])

        st.markdown(
            "The Perceptron convergence theorem guarantees convergence only for linearly "
            "separable data. Overlapping biomarker distributions between classes prevent the "
            "decision boundary from stabilizing."
        )

    with knn_tab:
        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown("**Vectorized Euclidean Distance**")
            st.latex(r"d(x, x^{(i)}) = \sqrt{\sum_{j=1}^d (x_j - x_j^{(i)})^2} = \|x - x^{(i)}\|_2")
            st.markdown("**Instance-Based Learning**")
            st.markdown(
                "KNN is a **non-parametric lazy learner**: it has no training phase and "
                "memorizes all training vectors. At inference time, distances to all training "
                "samples are computed to find the $k$ nearest neighbors."
            )
        with kc2:
            st.markdown("**Posterior Probability & Majority Vote**")
            st.latex(r"\hat{P}(Y=1 \mid x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y^{(i)}")
            st.latex(r"\hat{y} = \text{mode}\left(\{y^{(i)} : i \in \mathcal{N}_k(x)\}\right)")
            st.markdown(
                "We set $k=5$ (odd integer) to prevent tie-breaking ambiguity. "
                "KNN captures **non-linear local decision surfaces** without assuming "
                "a rigid global linear boundary."
            )

        st.markdown("---")

        # Test distance distribution
        @st.cache_data
        def _get_avg_5th_dists():
            return knn_model.kneighbors(X_test)[0][:, -1]

        avg_5th_dists = _get_avg_5th_dists()

        fig_knn_dist = go.Figure()
        fig_knn_dist.add_trace(go.Histogram(
            x=avg_5th_dists,
            nbinsx=25,
            marker=dict(color="#3DAF6B", line=dict(color="#1A1F2B", width=1), cornerradius=4),
            hovertemplate="Distance: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        fig_knn_dist.update_layout(plotly_layout(
            height=340,
            xaxis_title="Euclidean Distance to 5th Nearest Neighbor",
            yaxis_title="Test Patients",
            title=dict(text="Test Set Neighborhood Density Distribution (k=5)", font=dict(size=15)),
        ))
        st.plotly_chart(fig_knn_dist, use_container_width=True)

        km1, km2, km3 = st.columns(3)
        km1.metric("Hyperparameter (k)", "5")
        km2.metric("Distance Metric", "Euclidean (L2)")
        km3.metric("Training Memory", f"{len(X_train)} samples")

        st.markdown(
            "Feature standardization is critical for KNN: without Z-score normalization, "
            "high-magnitude enzymes (e.g. ALP up to 2110) would distort the Euclidean metric "
            "and ignore smaller markers like Bilirubin."
        )


# ══════════════════════════════════════════════════════════
# TAB 4 — EVALUATION
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Evaluation Results")
    st.caption(f"Test set: {len(y_test)} patients")
    st.markdown("")

    @st.cache_data
    def _get_evaluation_results():
        lr_p = lr_model.predict(X_test)
        perc_p = perc_model.predict(X_test)
        knn_p = knn_model.predict(X_test)
        return {
            "lr_rep": classification_report(y_test, lr_p),
            "perc_rep": classification_report(y_test, perc_p),
            "knn_rep": classification_report(y_test, knn_p),
            "cm_lr": confusion_matrix(y_test, lr_p),
            "cm_perc": confusion_matrix(y_test, perc_p),
            "cm_knn": confusion_matrix(y_test, knn_p),
        }

    eval_res = _get_evaluation_results()
    lr_rep = eval_res["lr_rep"]
    perc_rep = eval_res["perc_rep"]
    knn_rep = eval_res["knn_rep"]
    cm_lr = eval_res["cm_lr"]
    cm_perc = eval_res["cm_perc"]
    cm_knn = eval_res["cm_knn"]

    cm1, cm2, cm3 = st.columns(3, gap="medium")

    def make_cm_figure(cm, title, color_scale):
        labels = ["Healthy (0)", "Diseased (1)"]
        text_matrix = [[str(cm[i][j]) for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=color_scale,
            text=text_matrix, texttemplate="%{text}",
            textfont=dict(size=20, color="white"),
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig.update_layout(plotly_layout(
            height=300,
            title=dict(text=title, font=dict(size=14)),
            xaxis=dict(title="Predicted", gridcolor="rgba(0,0,0,0)", side="bottom"),
            yaxis=dict(title="Actual", gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        ))
        return fig

    with cm1:
        st.plotly_chart(
            make_cm_figure(cm_lr, "Logistic Regression", [[0, "#1A1F2B"], [1, ACCENT_BLUE]]),
            use_container_width=True,
        )
    with cm2:
        st.plotly_chart(
            make_cm_figure(cm_perc, "Perceptron", [[0, "#1A1F2B"], [1, ACCENT_ORANGE]]),
            use_container_width=True,
        )
    with cm3:
        st.plotly_chart(
            make_cm_figure(cm_knn, "KNN (k=5)", [[0, "#1A1F2B"], [1, ACCENT_GREEN]]),
            use_container_width=True,
        )

    st.markdown("---")

    # Scorecard
    st.markdown("##### Performance Comparison")
    metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
    keys = ["accuracy", "precision", "recall", "specificity", "f1_score"]
    lr_vals = [lr_rep[k] * 100 for k in keys]
    perc_vals = [perc_rep[k] * 100 for k in keys]
    knn_vals = [knn_rep[k] * 100 for k in keys]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="Logistic Regression", x=metric_names, y=lr_vals,
        marker=dict(color=ACCENT_BLUE, cornerradius=4),
        text=[f"{v:.1f}%" for v in lr_vals], textposition="outside",
        textfont=dict(size=11, color=TEXT_COLOR),
    ))
    fig_compare.add_trace(go.Bar(
        name="Perceptron", x=metric_names, y=perc_vals,
        marker=dict(color=ACCENT_ORANGE, cornerradius=4),
        text=[f"{v:.1f}%" for v in perc_vals], textposition="outside",
        textfont=dict(size=11, color=TEXT_COLOR),
    ))
    fig_compare.add_trace(go.Bar(
        name="KNN (k=5)", x=metric_names, y=knn_vals,
        marker=dict(color=ACCENT_GREEN, cornerradius=4),
        text=[f"{v:.1f}%" for v in knn_vals], textposition="outside",
        textfont=dict(size=11, color=TEXT_COLOR),
    ))
    fig_compare.update_layout(plotly_layout(
        height=420,
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 115], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12),
        ),
    ))
    st.plotly_chart(fig_compare, use_container_width=True)

    with st.expander("Metric Definitions"):
        fd1, fd2 = st.columns(2)
        with fd1:
            st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}")
            st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
            st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        with fd2:
            st.latex(r"\text{Specificity} = \frac{TN}{TN + FP}")
            st.latex(r"\text{F1} = 2 \cdot \frac{P \cdot R}{P + R}")

    st.markdown("---")
    st.markdown("##### Clinical Significance")
    cx1, cx2, cx3 = st.columns(3)
    with cx1:
        st.success(
            f"**Logistic Regression Recall: {lr_rep['recall']*100:.1f}%** — "
            f"Only {int(lr_rep['false_negatives'])} of "
            f"{int(lr_rep['true_positives'] + lr_rep['false_negatives'])} "
            f"diseased patients were missed."
        )
    with cx2:
        st.error(
            f"**Perceptron Recall: {perc_rep['recall']*100:.1f}%** — "
            f"{int(perc_rep['false_negatives'])} of "
            f"{int(perc_rep['true_positives'] + perc_rep['false_negatives'])} "
            f"diseased patients were misclassified."
        )
    with cx3:
        st.info(
            f"**KNN Specificity: {knn_rep['specificity']*100:.1f}%** — "
            f"Better balanced specificity ({int(knn_rep['true_negatives'])} true negatives) "
            f"demonstrating local non-linear clustering capability."
        )
    st.markdown(
        "In screening applications, false negatives carry higher cost than false positives. "
        "A missed diagnosis delays treatment; a false positive triggers a confirmatory test. "
        "Recall is therefore the primary optimization target."
    )


# ══════════════════════════════════════════════════════════
# TAB 5 — PREDICTION
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### Patient Risk Prediction")
    st.caption("Enter biomarker values or load a preset case.")

    PRESETS = {
        "Custom": None,
        "Healthy Profile": dict(
            age=32, gender="Female", total_bilirubin=0.8, direct_bilirubin=0.2,
            alkaline_phosphotase=160, alamine_aminotransferase=18,
            aspartate_aminotransferase=20, total_proteins=7.2, albumin=3.8,
            albumin_and_globulin_ratio=1.15,
        ),
        "Acute Liver Injury": dict(
            age=62, gender="Male", total_bilirubin=10.9, direct_bilirubin=5.5,
            alkaline_phosphotase=699, alamine_aminotransferase=64,
            aspartate_aminotransferase=100, total_proteins=7.5, albumin=3.2,
            albumin_and_globulin_ratio=0.74,
        ),
        "Borderline Case": dict(
            age=45, gender="Male", total_bilirubin=1.9, direct_bilirubin=0.9,
            alkaline_phosphotase=235, alamine_aminotransferase=45,
            aspartate_aminotransferase=55, total_proteins=6.7, albumin=3.0,
            albumin_and_globulin_ratio=0.80,
        ),
    }

    DEFAULT_PROFILE = dict(
        age=45, gender="Male", total_bilirubin=1.0, direct_bilirubin=0.3,
        alkaline_phosphotase=180, alamine_aminotransferase=25,
        aspartate_aminotransferase=30, total_proteins=7.0, albumin=3.5,
        albumin_and_globulin_ratio=1.0,
    )

    for k, v in DEFAULT_PROFILE.items():
        if f"p_{k}" not in st.session_state:
            st.session_state[f"p_{k}"] = v

    def compute_risk(p_age, p_gn, p_tb, p_db, p_alp, p_alt, p_ast, p_tp, p_ab, p_ag):
        vec = np.array([[p_age, p_gn, p_tb, p_db, p_alp, p_alt, p_ast, p_tp, p_ab, p_ag]], dtype=np.float64)
        svec = bundle["scaler"].transform(vec)
        lr_p = float(bundle["logistic_regression"].predict_proba(svec)[0])
        lr_c = int(lr_p >= 0.5)
        ps = float(bundle["perceptron"].decision_function(svec)[0])
        pc = 1 if ps >= 0 else 0
        knn_p = float(bundle["knn"].predict_proba(svec)[0])
        knn_c = int(knn_p >= 0.5)
        knn_dists, knn_idx = bundle["knn"].kneighbors(svec)
        return {
            "lr_p": lr_p,
            "lr_c": lr_c,
            "ps": ps,
            "pc": pc,
            "knn_p": knn_p,
            "knn_c": knn_c,
            "knn_dists": knn_dists,
            "svec": svec,
            "tb": p_tb,
            "db": p_db,
            "alp": p_alp,
            "alt": p_alt,
            "ast": p_ast,
            "tp": p_tp,
            "ab": p_ab,
            "ag": p_ag,
        }

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    def on_preset_change():
        choice = st.session_state.get("preset_selection", "Custom")
        preset_vals = PRESETS.get(choice)
        if preset_vals is not None:
            for pk, pv in preset_vals.items():
                st.session_state[f"p_{pk}"] = pv

    st.selectbox("Load Preset", list(PRESETS.keys()), key="preset_selection", on_change=on_preset_change, filter_mode=None)

    st.markdown("---")

    with st.form("risk_prediction_form", border=False):
        ic1, ic2, ic3 = st.columns(3, gap="large")

        with ic1:
            st.markdown("##### Demographics & Bilirubin")
            age = st.number_input("Age", 1, 100, key="p_age")
            gender = st.selectbox("Gender", ["Male", "Female"], key="p_gender")
            tb = st.number_input("Total Bilirubin (mg/dL)", 0.1, 75.0, step=0.1, key="p_total_bilirubin", help="Ref: 0.1–1.2")
            db = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 30.0, step=0.1, key="p_direct_bilirubin", help="Ref: 0.0–0.3")

        with ic2:
            st.markdown("##### Hepatic Enzymes")
            alp = st.number_input("ALP (IU/L)", 10, 2500, step=5, key="p_alkaline_phosphotase", help="Ref: 44–147")
            alt = st.number_input("ALT / SGPT (IU/L)", 5, 2000, step=2, key="p_alamine_aminotransferase", help="Ref: 7–56")
            ast = st.number_input("AST / SGOT (IU/L)", 5, 2500, step=2, key="p_aspartate_aminotransferase", help="Ref: 10–40")

        with ic3:
            st.markdown("##### Serum Proteins")
            tp = st.number_input("Total Proteins (g/dL)", 1.0, 15.0, step=0.1, key="p_total_proteins", help="Ref: 6.0–8.3")
            ab = st.number_input("Albumin (g/dL)", 0.5, 10.0, step=0.1, key="p_albumin", help="Ref: 3.4–5.4")
            ag = st.number_input("A/G Ratio", 0.1, 5.0, step=0.05, key="p_albumin_and_globulin_ratio", help="Ref: 0.8–2.0")

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Patient Risk Assessment", type="primary", use_container_width=True)

    results_placeholder = st.empty()

    if submitted:
        gn = 1 if gender == "Male" else 0
        st.session_state["last_prediction"] = compute_risk(
            age, gn, tb, db, alp, alt, ast, tp, ab, ag
        )

    res = st.session_state.get("last_prediction")
    if res is not None:
        with results_placeholder.container():
            st.markdown("---")
            r1, r2, r3 = st.columns(3, gap="medium")

            with r1:
                st.markdown("##### Logistic Regression")
                if res["lr_c"] == 1:
                    st.error(f"**High Risk** — {res['lr_p']*100:.1f}%")
                else:
                    st.success(f"**Low Risk** — {(1-res['lr_p'])*100:.1f}% healthy")

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["lr_p"] * 100,
                    number=dict(suffix="%", font=dict(size=24, color=TEXT_COLOR)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(size=10, color="#6B7280")),
                        bar=dict(color=ACCENT_RED if res["lr_c"] == 1 else ACCENT_GREEN, thickness=0.7),
                        bgcolor="rgba(255,255,255,0.04)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 50], color="rgba(61,175,107,0.1)"),
                            dict(range=[50, 100], color="rgba(217,74,74,0.1)"),
                        ],
                        threshold=dict(line=dict(color="#E8EAED", width=2), value=50, thickness=0.8),
                    ),
                ))
                fig_gauge.update_layout(
                    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                    font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                    height=200, margin=dict(l=20, r=20, t=20, b=10),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                with st.expander("Details"):
                    zv = float(np.dot(res["svec"], bundle["logistic_regression"].weights_)[0] + bundle["logistic_regression"].bias_)
                    st.latex(r"z = " + f"{zv:.4f}")
                    st.latex(r"\hat{{y}} = \sigma({:.4f}) = {:.4f}".format(zv, res["lr_p"]))

            with r2:
                st.markdown("##### Perceptron")
                if res["pc"] == 1:
                    st.error("**High Risk** — Class 1")
                else:
                    st.success("**Low Risk** — Class 0")

                fig_perc_ind = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=res["ps"],
                    number=dict(font=dict(size=30, color=TEXT_COLOR), prefix="z = "),
                    delta=dict(
                        reference=0, increasing=dict(color=ACCENT_RED),
                        decreasing=dict(color=ACCENT_GREEN),
                        font=dict(size=14),
                    ),
                ))
                fig_perc_ind.update_layout(
                    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                    font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                    height=200, margin=dict(l=20, r=20, t=30, b=10),
                )
                st.plotly_chart(fig_perc_ind, use_container_width=True)

                with st.expander("Details"):
                    dec = r"\geq 0 \Rightarrow 1" if res["pc"] == 1 else r"< 0 \Rightarrow 0"
                    st.latex(r"z = " + f"{res['ps']:.4f}")
                    st.latex(r"\text{{Step}}({:.4f}) {}".format(res["ps"], dec))

            with r3:
                st.markdown("##### KNN (k=5)")
                if res["knn_c"] == 1:
                    st.error(f"**High Risk** — {res['knn_p']*100:.0f}% votes")
                else:
                    st.success(f"**Low Risk** — {(1-res['knn_p'])*100:.0f}% healthy votes")

                fig_knn_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["knn_p"] * 100,
                    number=dict(suffix="%", font=dict(size=24, color=TEXT_COLOR)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(size=10, color="#6B7280")),
                        bar=dict(color=ACCENT_RED if res["knn_c"] == 1 else ACCENT_GREEN, thickness=0.7),
                        bgcolor="rgba(255,255,255,0.04)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 50], color="rgba(61,175,107,0.1)"),
                            dict(range=[50, 100], color="rgba(217,74,74,0.1)"),
                        ],
                        threshold=dict(line=dict(color="#E8EAED", width=2), value=50, thickness=0.8),
                    ),
                ))
                fig_knn_gauge.update_layout(
                    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                    font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                    height=200, margin=dict(l=20, r=20, t=20, b=10),
                )
                st.plotly_chart(fig_knn_gauge, use_container_width=True)

                with st.expander("Details"):
                    st.markdown(f"**Closest neighbor dist:** `{res['knn_dists'][0, 0]:.3f}`")
                    st.markdown(f"**Votes:** `{int(res['knn_p'] * 5)}/5` diseased neighbors")

            # Flags
            st.markdown("---")
            st.markdown("##### Biomarker Reference Check")
            flags = []
            if res["tb"] > 1.2: flags.append(("Total Bilirubin", f"{res['tb']:.1f}", "0.1–1.2", "Elevated"))
            if res["db"] > 0.3: flags.append(("Direct Bilirubin", f"{res['db']:.1f}", "0.0–0.3", "Elevated"))
            if res["alp"] > 147: flags.append(("ALP", f"{res['alp']}", "44–147", "Elevated"))
            if res["alt"] > 56: flags.append(("ALT", f"{res['alt']}", "7–56", "Elevated"))
            if res["ast"] > 40: flags.append(("AST", f"{res['ast']}", "10–40", "Elevated"))
            if res["ab"] < 3.4: flags.append(("Albumin", f"{res['ab']:.1f}", "3.4–5.4", "Low"))
            if res["ag"] < 0.8: flags.append(("A/G Ratio", f"{res['ag']:.2f}", "0.8–2.0", "Low"))

            if flags:
                st.warning(f"{len(flags)} biomarker(s) outside reference range:")
                st.table(pd.DataFrame(flags, columns=["Biomarker", "Value", "Reference", "Status"]))
            else:
                st.success("All biomarkers within reference intervals.")
    else:
        with results_placeholder.container():
            st.markdown("---")
            st.info("Select a preset or enter patient biomarker values above, then click **Run Patient Risk Assessment**.")


# ─── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Disclaimer:** JaunVer is an educational research project. All algorithms were "
    "implemented from scratch using NumPy. Not intended for clinical use."
)
