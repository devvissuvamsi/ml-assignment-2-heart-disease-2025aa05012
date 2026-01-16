
# Heart Disease Classification — ML Assignment 2

This repository contains an end‑to‑end machine learning project implementing six models and a Streamlit web application for heart‑disease prediction.

## Project Links
- GitHub Repository: https://github.com/devvissuvamsi/ml-assignment-2-heart-disease-2025aa05012
- Streamlit Web App: https://ml-assignment-2-heart-disease-2025aa05012.streamlit.app/

---
## a) Problem Statement
The goal of the assignment is to build and evaluate six different machine‑learning classification models to predict the presence of heart disease. A Streamlit application is required to demonstrate predictions, visualize metrics, and allow CSV uploads.

---
## b) Dataset Description
- Dataset Source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Target Variable: `target` (1 = heart disease, 0 = no heart disease)
- Total Features: 13
- Key Feature Types:
  - Numerical: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
  - Categorical: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
- Duplicate rows were removed during preprocessing to ensure fair model evaluation.

---
## c) Models & Evaluation Metrics
Six models were trained using a shared preprocessing pipeline:
- Logistic Regression
- Decision Tree
- K‑Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- XGBoost

Metrics evaluated for each model:
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

(Insert your final evaluation table here after running all models.)

---
## Observations (Short Notes)
(Insert brief observations about each model's behavior here.)

---
## Running the Project Locally
### 1. Clone the repository
```
git clone https://github.com/devvissuvamsi/ml-assignment-2-heart-disease-2025aa05012.git
cd ml-assignment-2-heart-disease-2025aa05012
```
### 2. Create a virtual environment (optional)
```
python -m venv venv
```
Activate it:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. (Optional) Train all models
```
python -m models.train_logreg
python -m models.train_decision_tree
python -m models.train_knn
python -m models.train_naive_bayes
python -m models.train_random_forest
python -m models.train_xgboost
```
### 5. Run the Streamlit app
```
streamlit run app.py
```

---
## Folder Structure
```
models/               # training scripts + saved model files (*.pkl)
utils/                # preprocessing, data loader, plotting, metrics
app.py                # Streamlit frontend
requirements.txt
README.md
```

---
## Summary
This project demonstrates preprocessing, model training, comparison, and deployment using Streamlit as required by Assignment 2.
