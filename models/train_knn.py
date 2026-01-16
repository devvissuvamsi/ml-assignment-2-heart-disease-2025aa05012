
# models/train_knn.py

import os
import sys
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score

from utils.preprocessing import build_preprocessor, TARGET_COL, NUM_COLS, CAT_COLS
from utils.data import load_and_clean_heart_csv  # <-- use the shared loader


DATA_PATH = os.path.join("data", "heart.csv")
MODEL_PATH = os.path.join("models", "knn_pipeline.pkl")


# -------------------------
# Safety & leakage checks
# -------------------------
def validate_feature_lists():
    errs = []
    if TARGET_COL in NUM_COLS:
        errs.append(f"{TARGET_COL} should not be in NUM_COLS")
    if TARGET_COL in CAT_COLS:
        errs.append(f"{TARGET_COL} should not be in CAT_COLS")
    inter = set(NUM_COLS).intersection(CAT_COLS)
    if inter:
        errs.append(f"Columns present in BOTH NUM_COLS and CAT_COLS: {sorted(inter)}")
    if errs:
        raise ValueError("Feature list validation failed:\n- " + "\n- ".join(errs))


def single_feature_perfect_predictors(df: pd.DataFrame) -> list:
    """
    Return features that alone perfectly separate the target (pre-split check).
    If any appears here, it's a strong leakage signal (or mislabeled/engineered column).
    """
    bad = []
    for col in NUM_COLS + CAT_COLS:
        g = df.groupby(col)[TARGET_COL].nunique()
        # Every group has exactly one target value (and there's >1 group)
        if g.max() == 1 and g.min() == 1 and len(g) > 1:
            bad.append(col)
    return bad


def build_model() -> KNeighborsClassifier:
    # Keep it simple and similar to your DT script philosophy (no grid here)
    return KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        p=2,
        metric="minkowski"
    )


def audit_zero_distance_overlap(preprocessor, X_train, X_val) -> int:
    """
    Fit preprocessor on X_train, transform both splits, and count how many
    validation rows have zero distance to at least one training row.
    If this equals len(X_val), KNN can easily become perfect.
    """
    Xt = preprocessor.fit_transform(X_train)
    Xv = preprocessor.transform(X_val)

    # Work with dense arrays for distance check (only for audit on validation size)
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    Xv_dense = Xv.toarray() if hasattr(Xv, "toarray") else Xv

    def row_hash(mat):
        # Round to mitigate tiny float noise from scaling
        return {tuple(np.round(row, 8)) for row in mat}

    train_hashes = row_hash(Xt_dense)
    zero_d = sum(1 for row in Xv_dense if tuple(np.round(row, 8)) in train_hashes)
    return zero_d


def main():
    validate_feature_lists()

    # 1) Load & clean (shared method across all models)
    df = load_and_clean_heart_csv(DATA_PATH, TARGET_COL, NUM_COLS, CAT_COLS)

    # 2) Optional leakage probe: single-feature perfect predictors
    suspicious = single_feature_perfect_predictors(df)
    if suspicious:
        print(f"[ALERT] Single-feature perfect predictors found: {suspicious}")
        print("        These columns by themselves deterministically map to target.")
        print("        This is a strong sign of leakage or a mislabeled/engineered column.")
        # Consider dropping them here if they truly are leaked.

    # 3) Split
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL].astype(int)
    print("Class balance:", y.value_counts(normalize=True).to_dict())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Build pipeline
    preprocessor = build_preprocessor()
    model = build_model()

    # ---- Audit: zero-distance overlap AFTER preprocessing (fit on TRAIN only) ----
    zero_overlap = audit_zero_distance_overlap(build_preprocessor(), X_train, X_val)
    if zero_overlap == len(X_val):
        print(f"[ALERT] Every validation row has a zero-distance match in TRAIN after preprocessing "
              f"({zero_overlap}/{len(X_val)}). Expect KNN to be (nearly) perfect.")
        print("        This usually means many repeated feature vectors or a near-deterministic column.")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # 5) Fit
    pipeline.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = pipeline.predict(X_val)

    print("K-Nearest Neighbors (KNN) – Validation Report")
    print(classification_report(y_val, y_pred, digits=4))

    try:
        if y_val.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            print(f"AUC: {auc:.4f}")
    except Exception:
        pass

    # 7) Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(pipeline, MODEL_PATH)
    print(f"Saved KNN pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
