
# models/train_xgboost.py

import os
import sys
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

from utils.preprocessing import build_preprocessor, TARGET_COL, NUM_COLS, CAT_COLS
from utils.data import load_and_clean_heart_csv  # centralized loader with de-dup

DATA_PATH = os.path.join("data", "train.csv")
MODEL_PATH = os.path.join("models", "artifacts",  "xgboost_pipeline.pkl")


def build_model() -> XGBClassifier:
    """
    A strong, stable baseline for tabular classification using XGBoost.
    - tree_method='hist' for speed
    - eval_metric='logloss' to suppress warnings and keep metrics consistent
    - n_estimators/learning_rate chosen for a balanced bias-variance trade-off
    """
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
        # Ensure consistent binary handling
        objective="binary:logistic",
        use_label_encoder=False
    )


def main():
    # 1) Load once, cleaned & deduplicated (same for all models)
    df = load_and_clean_heart_csv(DATA_PATH, TARGET_COL, NUM_COLS, CAT_COLS)

    # 2) Split features/target
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Build pipeline (preprocessing inside the pipeline)
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", build_model())
    ])

    # 4) Fit
    pipeline.fit(X_train, y_train)

    # 5) Evaluate on validation
    y_pred = pipeline.predict(X_val)

    # AUC if predict_proba is available and both classes present
    try:
        if y_val.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
        else:
            auc = None
    except Exception:
        auc = None

    print("XGBoost – Validation Report")
    print(classification_report(y_val, y_pred, digits=4))
    if auc is not None:
        print(f"AUC: {auc:.4f}")

    # 6) Persist the pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(pipeline, MODEL_PATH)
    print(f"Saved XGBoost pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
