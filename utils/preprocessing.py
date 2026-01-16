
# utils/preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

TARGET_COL = "target"

NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

def build_preprocessor():
    """
    Builds and returns a ColumnTransformer that:
    - Standardizes numerical features
    - One-hot encodes categorical features
    - Forces DENSE output (so GaussianNB doesn't need a to_dense transformer)
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Compatibility: scikit-learn >=1.2 uses 'sparse_output', older uses 'sparse'
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback for older versions
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("onehot", ohe)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ],
        # Force dense even if parts are sparse
        sparse_threshold=0.0,
    )

    return preprocessor
