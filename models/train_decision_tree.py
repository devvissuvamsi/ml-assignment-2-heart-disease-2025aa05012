# models/train_decision_tree.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from joblib import dump

from utils.preprocessing import build_preprocessor, TARGET_COL, NUM_COLS, CAT_COLS


DATA_PATH = os.path.join("data", "heart.csv")
MODEL_PATH = os.path.join("models", "decision_tree_pipeline.pkl")


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET_COL])
    df = df.dropna(subset=NUM_COLS + CAT_COLS)
    return df


def build_model():
    return DecisionTreeClassifier(
        criterion="entropy",   # information gain
        random_state=42
    )


def main():
    df = load_data(DATA_PATH)

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    model = build_model()

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    print("Decision Tree – Validation Report")
    print(classification_report(y_val, y_pred, digits=4))

    dump(pipeline, MODEL_PATH)
    print(f"Saved Decision Tree pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
