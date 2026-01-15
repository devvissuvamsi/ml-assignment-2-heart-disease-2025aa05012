# utils/preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Adjust these if your dataset columns change
TARGET_COL = "target"

NUM_COLS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CAT_COLS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


def build_preprocessor():
    """
    Builds and returns a ColumnTransformer that:
    - Standardizes numerical features
    - One-hot encodes categorical features
    """

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS)
        ]
    )

    return preprocessor
