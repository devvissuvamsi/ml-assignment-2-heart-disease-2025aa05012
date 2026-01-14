import streamlit as st
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline
from utils.metrics import compute_metrics
from utils.plotting import plot_confusion_matrix, plot_roc

st.set_page_config(page_title="Heart Disease Classification", layout="wide")

MODEL_PATHS = {
    "Logistic Regression": "models/logreg_model.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "kNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
}
PREPROCESSOR_PATH = "models/preprocessor.pkl"

TARGET_COL = "target"
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

from utils.plotting import plot_confusion_matrix, plot_roc

@st.cache_resource
def load_preprocessor():
    return load(PREPROCESSOR_PATH)

def load_model(model_name):
    model = load(MODEL_PATHS[model_name])
    preprocessor = load_preprocessor()
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

st.title("💓 Heart Disease Classification")
st.write("Upload a CSV file and choose a model to evaluate.")

model_choice = st.selectbox("Choose a model", list(MODEL_PATHS.keys()))
pipe = load_model(model_choice)

uploaded = st.file_uploader("Upload test CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in (NUM_COLS + CAT_COLS + [TARGET_COL]) if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        X_test = df[NUM_COLS + CAT_COLS]
        y_test = df[TARGET_COL].astype(int)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_proba)

        # Display metrics in a nice grid
        st.subheader(f"📊 Evaluation Metrics — {model_choice}")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['precision']:.3f}")
        col3.metric("Recall", f"{metrics['recall']:.3f}")
        col4.metric("F1 Score", f"{metrics['f1']:.3f}")
        col5.metric("MCC", f"{metrics['mcc']:.3f}")
        if metrics['auc'] is not None:
            col6.metric("AUC", f"{metrics['auc']:.3f}")
        else:
            col6.metric("AUC", "N/A")

        # Visualizations in expanders
        with st.expander("Confusion Matrix"):
            st.pyplot(plot_confusion_matrix(y_test, y_pred))

        if y_proba is not None:
            with st.expander("ROC Curve"):
                st.pyplot(plot_roc(y_test, y_proba))
