
# models/train_naive_bayes.py

import os
import sys
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score

from utils.preprocessing import build_preprocessor, TARGET_COL, NUM_COLS, CAT_COLS
from utils.data import load_and_clean_heart_csv

DATA_PATH = os.path.join("data", "train.csv")
MODEL_PATH = os.path.join("models", "artifacts",  "naive_bayes_pipeline.pkl")

def build_model() -> GaussianNB:
    return GaussianNB(var_smoothing=1e-9)

def main():
    df = load_and_clean_heart_csv(DATA_PATH, TARGET_COL, NUM_COLS, CAT_COLS)

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", build_model())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    try:
        y_proba = pipeline.predict_proba(X_val)[:, 1] if y_val.nunique() == 2 else None
        auc = roc_auc_score(y_val, y_proba) if y_proba is not None else None
    except Exception:
        auc = None

    print("Naive Bayes (GaussianNB) – Validation Report")
    print(classification_report(y_val, y_pred, digits=4))
    if auc is not None:
        print(f"AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(pipeline, MODEL_PATH)
    print(f"Saved Naive Bayes pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
