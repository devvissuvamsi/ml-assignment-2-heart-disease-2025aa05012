
# models/train_logreg.py

import os
import sys
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from utils.preprocessing import build_preprocessor, TARGET_COL, NUM_COLS, CAT_COLS
from utils.data import load_and_clean_heart_csv  # <-- centralized loader with de-dup

DATA_PATH = os.path.join("data", "heart.csv")
MODEL_PATH = os.path.join("models", "logreg_pipeline.pkl")


def build_model() -> LogisticRegression:
    # Using lbfgs with balanced classes is a good default for this dataset size
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )


def main():
    # 1) Load once, cleaned & deduplicated (consistent across all models)
    df = load_and_clean_heart_csv(DATA_PATH, TARGET_COL, NUM_COLS, CAT_COLS)

    # 2) Split features/target
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Build pipeline (preprocessing encapsulated inside)
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", build_model())
    ])

    # 4) Fit
    pipeline.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = pipeline.predict(X_val)
    # AUC if predict_proba available and binary target present
    try:
        if y_val.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
        else:
            auc = None
    except Exception:
        auc = None

    print("Logistic Regression – Validation Report")
    print(classification_report(y_val, y_pred, digits=4))
    if auc is not None:
        print(f"AUC: {auc:.4f}")

    # 6) Persist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(pipeline, MODEL_PATH)
    print(f"Saved Logistic Regression pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
