
# Heart Disease Classification

This repository contains an end‑to‑end machine learning project that trains six classification models to predict heart disease and provides a simple Streamlit application for testing and evaluation.

## Project Links
- GitHub Repository: https://github.com/devvissuvamsi/ml-assignment-2-heart-disease-2025aa05012
- Streamlit App: https://ml-assignment-2-heart-disease-2025aa05012.streamlit.app/
- Dataset Source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## Overview
The goal of this project is to train multiple machine learning models on the Kaggle Heart Disease dataset and compare their performance. A Streamlit app is included to let users evaluate models on the built‑in dataset or upload their own CSV file.

## Models Implemented
- Logistic Regression
- Decision Tree
- K‑Nearest Neighbors (KNN)
- Naive Bayes (GaussianNB)
- Random Forest
- XGBoost

All models share a consistent preprocessing pipeline and use a cleaned, deduplicated version of the dataset before splitting.

## Running This Project Locally
Follow these steps to download and run the project on your machine:

### 1. Clone the GitHub repository
```
git clone https://github.com/devvissuvamsi/ml-assignment-2-heart-disease-2025aa05012.git
cd ml-assignment-2-heart-disease-2025aa05012
```

### 2. Create a virtual environment (optional but recommended)
```
python -m venv venv
```
Activate it:
- Windows:
```
venv\Scriptsctivate
```
- macOS/Linux:
```
source venv/bin/activate
```

### 3. Install required dependencies
```
pip install -r requirements.txt
```

### 4. Train Models (optional)
Each model has its own training script:
```
python -m models.train_logreg
python -m models.train_decision_tree
python -m models.train_knn
python -m models.train_naive_bayes
python -m models.train_random_forest
python -m models.train_xgboost
```
This will generate their `.pkl` files inside the `models/` directory.

### 5. Run the Streamlit Application
```
streamlit run app.py
```
This will open the web app in your browser.

## Streamlit App Features
- Select and evaluate any of the six models
- View key metrics and visualizations
- Upload a custom CSV file for prediction
- Inspect dataset overview

## Folder Structure
```
models/               # training scripts + saved model pipelines
utils/                # preprocessing, data loading, metrics, plotting
app.py                # Streamlit application
requirements.txt
README.md
```

## Summary
This repository demonstrates a clean workflow for data preprocessing, model training, evaluation, and deployment using Streamlit.
