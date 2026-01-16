
# Heart Disease Classification

A clean, minimal end-to-end ML project implementing 6 classification models and a Streamlit app for heart disease prediction.

## 🔗 Links
- **GitHub Repo Url:** https://github.com/devvissuvamsi/ml-assignment-2-heart-disease-2025aa05012
- **Streamlit App Url:** https://ml-assignment-2-heart-disease-2025aa05012.streamlit.app/
- **Dataset:** Kaggle Heart Disease Dataset (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## 📌 Overview
This project builds and compares six machine learning models on the Kaggle Heart Disease dataset and provides an interactive Streamlit app for evaluation and predictions.

## 📊 Models Implemented
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

All models share a common preprocessing pipeline and use a deduplicated dataset for fair comparison.

## 📈 Evaluation Metrics
For each model, the following metrics are computed:
Accuracy, AUC, Precision, Recall, F1 Score, MCC

(Add your final comparison table here)

## 🚀 Streamlit App Features
- Model selection
- Built‑in dataset evaluation
- CSV upload for predictions
- Confusion Matrix & ROC Curve
- Compact performance summary

## 📁 Project Structure
```
models/        # training scripts + saved pipelines
utils/         # preprocessing, metrics, plotting, data loader
app.py         # Streamlit frontend
data/heart.csv
README.md
```

## ▶️ Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```

## ✨ Summary
A clean ML workflow demonstrating preprocessing, model training, comparison, and deployment using Streamlit.
