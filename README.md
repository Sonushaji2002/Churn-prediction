Telco Customer Churn Prediction


📌 Overview
This project predicts customer churn in a telecommunications company using machine learning and a responsive Streamlit web application. It helps stakeholders proactively identify at-risk customers and optimize retention strategies.

📁 Dataset
Source: Kaggle - Telco Customer Churn Dataset

Features Used: Customer demographics, service details, tenure, billing

Target: Churn (Yes/No)

🧠 Problem Statement
Churn leads to major revenue losses for telecom businesses. This project:

Analyzes customer patterns and service usage

Predicts likelihood of churn using classification models

Empowers data-driven decision-making to improve customer satisfaction and loyalty

🔍 Features
✅ Interactive input of customer data via Streamlit UI

✅ Preprocessing includes:

Label encoding for binary categories

One-hot encoding for categorical features

Feature scaling using a pre-trained scaler

✅ Trained model using Random Forest

✅ Imbalanced data handled with SMOTE

✅ Real-time prediction using .sav model file and .pkl metadata

✅ Custom-styled UI with background image and result boxes

🛠️ Tech Stack

Category	 Tools/Libraries
Language	 Python
Interface	 Streamlit
ML Model	 Random Forest Classifier
Preprocessing	 pandas, scikit-learn, imbalanced-learn
Model Saving	 pickle (.sav, .pkl)
UI Styling	 Custom CSS in Streamlit
Version Control 	Git, GitHub

📊 Model Performance
Model: Random Forest Classifier

Accuracy: ~80% (with SMOTE applied)

Metrics: Accuracy, Precision, Recall, F1-Score

🤝 Acknowledgments
Kaggle Telco Churn Dataset

imbalanced-learn for SMOTE

Streamlit for the interactive web app framework

