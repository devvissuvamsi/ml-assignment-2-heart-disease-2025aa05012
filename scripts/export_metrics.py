
#!/usr/bin/env python3
"""
Export metrics for all saved models and (optionally) update README.md tables.

Usage:
  python scripts/export_metrics.py [--update-readme]

What it does:
  • Loads data/heart.csv and the six saved model pipelines (*.pkl)
  • Computes: Accuracy, AUC, Precision, Recall, F1, MCC for each model
  • Writes:
      metrics/metrics_summary.csv
      metrics/metrics_table.md
      metrics/observations_table.md
  • If --update-readme is given, replaces the content between the markers in README.md:
      <!-- METRICS_TABLE_START --> ... <!-- METRICS_TABLE_END -->
      <!-- OBS_TABLE_START --> ... <!-- OBS_TABLE_END -->
"""


# --- Ensure repo root is on sys.path so "utils" can be imported ---
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import argparse
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)

# Project-specific imports (feature lists and target)
try:
    from utils.preprocessing import TARGET_COL, NUM_COLS, CAT_COLS
except Exception as e:
    print('[ERROR] Could not import utils.preprocessing:', e)
    sys.exit(1)

# Paths to the six trained pipelines
MODELS = [
    ("Logistic Regression", "models/logreg_pipeline.pkl"),
    ("Decision Tree",       "models/decision_tree_pipeline.pkl"),
    ("KNN",                 "models/knn_pipeline.pkl"),
    ("Naive Bayes",         "models/naive_bayes_pipeline.pkl"),
    ("Random Forest",       "models/random_forest_pipeline.pkl"),
    ("XGBoost",             "models/xgboost_pipeline.pkl"),
]

OUT_DIR = "metrics"
os.makedirs(OUT_DIR, exist_ok=True)


def safe_predict_proba(pipe, X):
    """Return probabilities for positive class if available; otherwise a sigmoid over decision_function
    or fallback to predictions as float."""
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    else:
        preds = pipe.predict(X)
        return preds.astype(float)


def compute_for_model(name, path, X, y):
    row = {
        "Model": name,
        "Accuracy": None, "AUC": None, "Precision": None,
        "Recall": None, "F1": None, "MCC": None
    }
    if not os.path.exists(path):
        print(f"[WARN] Missing model file: {path}")
        return row

    try:
        pipe = load(path)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return row

    try:
        y_pred = pipe.predict(X)
        y_proba = safe_predict_proba(pipe, X)

        row["Accuracy"]  = float(accuracy_score(y, y_pred))
        row["Precision"] = float(precision_score(y, y_pred, zero_division=0))
        row["Recall"]    = float(recall_score(y, y_pred, zero_division=0))
        row["F1"]        = float(f1_score(y, y_pred, zero_division=0))
        row["MCC"]       = float(matthews_corrcoef(y, y_pred))

        try:
            row["AUC"] = float(roc_auc_score(y, y_proba)) if y.nunique() == 2 else None
        except Exception:
            row["AUC"] = None
    except Exception as e:
        print(f"[WARN] Could not compute metrics for {name}: {e}")

    return row


def to_markdown_table(df):
    # Ensure correct column order
    cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    df2 = df[cols].copy()

    # Round numeric columns for readability
    for c in cols[1:]:
        df2[c] = df2[c].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else ("" if x is None else x))

    header = (
        "| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df2.values]
    return header + "\n" + "\n".join(rows)


def observations_from_metrics(df):
    """Generate a compact observation for each model based on AUC/Precision/Recall/F1."""
    obs = []
    for _, r in df.iterrows():
        name = r["Model"]
        auc  = r["AUC"]
        prec = r["Precision"]
        rec  = r["Recall"]
        f1   = r["F1"]

        note = []
        if pd.notna(auc):
            if auc >= 0.90:
                note.append("Strong discriminative ability (high AUC).")
            elif auc >= 0.80:
                note.append("Good discriminative ability.")
            else:
                note.append("Moderate discriminative ability.")

        if pd.notna(prec) and pd.notna(rec):
            if rec > prec + 0.05:
                note.append("Recall > Precision (leans towards sensitivity).")
            elif prec > rec + 0.05:
                note.append("Precision > Recall (leans towards specificity).")
            else:
                note.append("Balanced precision/recall.")

        if pd.notna(f1) and f1 >= 0.85:
            note.append("Strong overall F1.")

        obs.append({"Model": name, "Observation": " ".join(note)})
    return pd.DataFrame(obs)


def update_readme(metrics_md, obs_md):
    """Replace README blocks between markers with fresh tables; if markers missing, append at end."""
    path = "README.md"
    if not os.path.exists(path):
        print("[WARN] README.md not found; skipping update.")
        return

    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    def replace_block(text, start_tag, end_tag, new_content):
        if start_tag in text and end_tag in text:
            s = text.index(start_tag) + len(start_tag)
            e = text.index(end_tag)
            return text[:s] + "\n" + new_content + "\n" + text[e:]
        else:
            # Append markers + block at end if not present
            return text + f"\n\n{start_tag}\n{new_content}\n{end_tag}\n"

    txt = replace_block(txt, "<!-- METRICS_TABLE_START -->", "<!-- METRICS_TABLE_END -->", metrics_md)
    txt = replace_block(txt, "<!-- OBS_TABLE_START -->", "<!-- OBS_TABLE_END -->", obs_md)

    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print("README.md updated with metrics and observations.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-readme", action="store_true", help="Replace tables inside README.md markers")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv("data/heart.csv")
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL].astype(int)

    # Compute metrics for each model
    rows = [compute_for_model(n, p, X, y) for n, p in MODELS]
    res = pd.DataFrame(rows)

    # Save raw CSV summary
    res.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

    # Build Markdown tables
    metrics_md = to_markdown_table(res)
    obs_df = observations_from_metrics(res)
    obs_md = "| Model | Observation |\n|---|---|\n" + "\n".join(
        [f"| {r['Model']} | {r['Observation']} |" for _, r in obs_df.iterrows()]
    )

    # Write MD artifacts
    with open(os.path.join(OUT_DIR, "metrics_table.md"), "w", encoding="utf-8") as f:
        f.write(metrics_md)
    with open(os.path.join(OUT_DIR, "observations_table.md"), "w", encoding="utf-8") as f:
        f.write(obs_md)

    # Echo to console
    print("\n--- Metrics Table (Markdown) ---\n")
    print(metrics_md)
    print("\n--- Observations Table (Markdown) ---\n")
    print(obs_md)

    # Update README if requested
    if args.update_readme:
        update_readme(metrics_md, obs_md)


if __name__ == "__main__":
    main()
