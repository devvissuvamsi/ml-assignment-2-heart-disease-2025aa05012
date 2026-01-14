import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

DATA_PATH = os.path.join('data', 'heart.csv')
MODEL_PATH = os.path.join('models', 'logreg_model.pkl')
PREPROCESSOR_PATH = os.path.join('models', 'preprocessor.pkl')

# Adjust these based on your actual CSV columns
TARGET_COL = 'target'
NUM_COLS = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
]
CAT_COLS = [
    'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
]

def load_data(path):
    df = pd.read_csv(path)
    # Basic cleaning: drop rows with missing target or features
    df = df.dropna(subset=[TARGET_COL])
    df = df.dropna(subset=NUM_COLS + CAT_COLS)
    return df

def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_COLS),
            ('cat', categorical_transformer, CAT_COLS)
        ]
    )
    return preprocessor

def build_model():
    # Balanced class weights help if target is imbalanced
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
    return clf

def main():
    df = load_data(DATA_PATH)
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    model = build_model()

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]

    print('Validation classification report:')
    print(classification_report(y_val, y_pred, digits=4))

    preprocessor.fit(X_train)
    dump(preprocessor, PREPROCESSOR_PATH)
    dump(model, MODEL_PATH)

    print(f'Saved preprocessor to {PREPROCESSOR_PATH}')
    print(f'Saved model to {MODEL_PATH}')

if __name__ == '__main__':
    main()
