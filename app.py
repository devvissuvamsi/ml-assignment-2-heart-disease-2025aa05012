
import streamlit as st
import pandas as pd
from joblib import load

from utils.metrics import compute_metrics
from utils.plotting import plot_confusion_matrix, plot_roc

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

# Compact styling: tighter paddings + compact plots
st.markdown(
    """
    <style>
        /* Reduce padding to make sections look tighter */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        /* Compact metrics row: make st.metric look like badges */
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem; /* slightly smaller */
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #9aa0a6;
        }

        /* Reduce table row height */
        .stDataFrame table tbody tr td {
            padding-top: 0.3rem;
            padding-bottom: 0.3rem;
        }

        /* Slightly smaller plot containers */
        .stPlot {
            transform: scale(0.90);
            transform-origin: top left;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💓 Heart Disease Classification System")
st.caption("Machine Learning models for predicting heart disease")

# =========================================================
# Constants
# =========================================================
DATA_PATH = "data/heart.csv"
TARGET_COL = "target"

FEATURE_COLS = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

MODEL_PATHS = {
    "Logistic Regression": "models/logreg_pipeline.pkl",
    "Decision Tree": "models/decision_tree_pipeline.pkl",
}

# =========================================================
# Cached loader
# =========================================================
@st.cache_resource
def load_pipeline(path: str):
    return load(path)

# =========================================================
# Helper: Metrics + visuals + summary (COMPACT)
# =========================================================
def render_results(model_name: str, y_true, y_pred, y_proba):
    """
    Renders:
      1) Compact KPI metrics row
      2) Compact visuals (confusion matrix + ROC) side-by-side
      3) Full metrics table in an expander
      4) One-line summary
    """
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # ---------- Compact KPI metrics (6 tiles) ----------
    st.markdown("### 📈 Evaluation")
    kpi_cols = st.columns(6)

    # Round once for display
    acc = round(metrics["accuracy"], 3)
    prec = round(metrics["precision"], 3)
    rec = round(metrics["recall"], 3)
    f1 = round(metrics["f1"], 3)
    auc = round(metrics["auc"], 3)
    mcc = round(metrics["mcc"], 3)

    # Show KPIs
    kpi_cols[0].metric("Accuracy", f"{acc:.3f}")
    kpi_cols[1].metric("Precision", f"{prec:.3f}")
    kpi_cols[2].metric("Recall", f"{rec:.3f}")
    kpi_cols[3].metric("F1 Score", f"{f1:.3f}")
    kpi_cols[4].metric("AUC", f"{auc:.3f}")
    kpi_cols[5].metric("MCC", f"{mcc:.3f}")

    # ---------- Compact visualizations ----------
    st.markdown("### 🔍 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Confusion Matrix")
        # Get fig and shrink it to be compact
        fig_cm = plot_confusion_matrix(y_true, y_pred)
        try:
            # If fig is a matplotlib Figure, we can resize
            fig_cm.set_size_inches(3.6, 3.2)  # width, height in inches
        except Exception:
            pass
        st.pyplot(fig_cm, use_container_width=True)

    with col2:
        st.caption("ROC Curve")
        fig_roc = plot_roc(y_true, y_proba)
        try:
            fig_roc.set_size_inches(3.6, 3.2)
        except Exception:
            pass
        st.pyplot(fig_roc, use_container_width=True)

    # ---------- One-line summary ----------
    st.markdown("### 🧠 Model Performance Summary")
    summary = (
        f"The **{model_name}** model achieves **{acc:.2f}** accuracy, "
        f"with precision **{prec:.2f}** and recall **{rec:.2f}**, "
        f"indicating stable classification performance."
    )
    st.info(summary)


def _safe_predict_proba(pipe, X):
    """
    Returns proba for positive class if available; otherwise tries decision_function
    and normalizes via a sigmoid fallback.
    """
    import numpy as np

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        # handle binary shape consistently: pick positive class column if present
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        else:
            # if model returns single column (rare), use that
            return proba.reshape(-1)
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        # sigmoid to map to (0,1)
        return 1 / (1 + np.exp(-scores))
    else:
        # fallback: use predictions as pseudo-proba (0/1)
        preds = pipe.predict(X)
        return preds.astype(float)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Built-in Dataset Evaluation",
    "📂 Upload New Dataset",
    "ℹ️ Dataset Overview"
])

# =========================================================
# TAB 1 — Built-in dataset
# =========================================================
with tab1:
    st.subheader("Evaluation using Built-in Heart Disease Dataset")

    model_choice = st.selectbox(
        "Select Model",
        list(MODEL_PATHS.keys()),
        key="tab1_model"
    )

    pipe = load_pipeline(MODEL_PATHS[model_choice])

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    y_pred = pipe.predict(X)
    y_proba = _safe_predict_proba(pipe, X)

    render_results(model_choice, y, y_pred, y_proba)

# =========================================================
# TAB 2 — Uploaded dataset
# =========================================================
with tab2:
    st.subheader("Evaluate Model on Uploaded Dataset")

    model_choice = st.selectbox(
        "Select Model",
        list(MODEL_PATHS.keys()),
        key="tab2_model"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file (must include target column)",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            pipe = load_pipeline(MODEL_PATHS[model_choice])

            X = df[FEATURE_COLS]
            y = df[TARGET_COL].astype(int)

            y_pred = pipe.predict(X)
            y_proba = _safe_predict_proba(pipe, X)

            render_results(model_choice, y, y_pred, y_proba)

# =========================================================
# TAB 3 — Dataset overview
# =========================================================
with tab3:
    st.subheader("Heart Disease Dataset Overview")

    st.markdown("""
**Dataset Source:** UCI Heart Disease Dataset (via Kaggle)  

**Problem Type:** Binary Classification  

**Target Variable:** `target`  
- `1` → Presence of heart disease  
- `0` → Absence of heart disease  

**Total Features:** 13  

### Feature Categories

**Numerical Features**
- Age  
- Resting Blood Pressure (`trestbps`)  
- Serum Cholesterol (`chol`)  
- Maximum Heart Rate (`thalach`)  
- ST Depression (`oldpeak`)  

**Categorical Features**
- Sex  
- Chest Pain Type (`cp`)  
- Fasting Blood Sugar (`fbs`)  
- Resting ECG (`restecg`)  
- Exercise Induced Angina (`exang`)  
- ST Slope (`slope`)  
- Number of Major Vessels (`ca`)  
- Thalassemia (`thal`)  

### Objective
To predict the presence of heart disease using clinical and demographic
features through machine learning models.
""")
