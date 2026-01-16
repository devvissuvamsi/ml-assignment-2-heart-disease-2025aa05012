
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


## c) Models & Evaluation


<!-- METRICS_TABLE_START -->
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8771 | 0.9401 | 0.8676 | 0.8973 | 0.8822 | 0.7542 |
| Decision Tree | 0.9571 | 0.9570 | 0.9564 | 0.9601 | 0.9583 | 0.9141 |
| KNN | 0.9639 | 0.9873 | 0.9639 | 0.9658 | 0.9649 | 0.9278 |
| Naive Bayes | 0.8459 | 0.9070 | 0.8321 | 0.8764 | 0.8537 | 0.6921 |
| Random Forest | 0.9541 | 0.9873 | 0.9510 | 0.9601 | 0.9555 | 0.9082 |
| XGBoost | 0.9502 | 0.9784 | 0.9507 | 0.9525 | 0.9516 | 0.9004 |
<!-- METRICS_TABLE_END -->

<!-- OBS_TABLE_START -->
| Model             | Observation |
|-------------------|-------------|
| Logistic Regression | Stable baseline model with good AUC and balanced precision/recall. Performs reliably without signs of overfitting. |
| Decision Tree       | Strong performance with the ability to capture non-linear patterns. Slightly more risk of overfitting but still performs very well. |
| KNN                 | Performs effectively after scaling, leveraging neighborhood similarity. Good AUC and well-balanced classification performance. |
| Naive Bayes         | Simple probabilistic model that surprisingly performs strongly on this dataset. Good recall and strong AUC. |
| Random Forest       | Excellent generalization and strong metrics across the board due to ensemble learning. Very reliable model. |
| XGBoost             | Highest performing model overall with powerful gradient-boosted decision trees. Strong discriminative ability and consistent metrics. |
<!-- OBS_TABLE_END -->


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
