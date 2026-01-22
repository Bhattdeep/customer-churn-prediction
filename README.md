Customer Churn Prediction Using Machine Learning
📌 Overview

Customer churn is a critical challenge in the telecom industry, where customers discontinue services due to factors such as pricing, service quality, or competition.
This project builds a machine learning–based churn prediction system using a real-world telecom dataset to identify customers who are likely to churn, enabling proactive retention strategies.

🎯 Objective

To develop and evaluate machine learning models that predict customer churn based on customer demographics, service usage, and account-related information.

📊 Dataset

Dataset Name: Telco Customer Churn Dataset
Source: IBM Sample Data
File: WA_Fn-UseC_-Telco-Customer-Churn.csv
Total Records: 7,043 customer
Features Include:
Customer demographics (gender, senior citizen, partner, dependents)
Service subscriptions (internet service, online security, streaming services, etc.)
Account information (contract type, tenure, payment method)
Target Variable: Churn (Yes / No)

🛠️ Technologies Used

Python
Pandas & NumPy – data preprocessing and manipulation
Matplotlib & Seaborn – data visualization and EDA
Scikit-learn – model training and evaluation
Imbalanced-learn (SMOTE) – handling class imbalance
XGBoost – advanced ensemble learning
Jupyter Notebook
VS Code

⚙️ Project Workflow

Loaded and inspected the telecom churn dataset
Cleaned and preprocessed data (handled missing and inconsistent values)
Encoded categorical variables
Handled class imbalance using SMOTE
Trained multiple machine learning models
Evaluated models using accuracy, precision, recall, and F1-score
Saved the best-performing model for future use

🤖 Machine Learning Models Used

Decision Tree Classifier
Random Forest Classifier
XGBoost Classifier (Final Selected Model)

✅ Results

Achieved an F1-score of approximately 0.78
XGBoost provided the best overall performance
Balanced precision and recall for churn prediction
Model effectively identifies customers at high risk of churn

📁 Project Structure
customer-churn-prediction/
│
├── env/                               # Virtual environment
│
├── churn.py                           # Main Python script for training & evaluation
├── Customer_Churn_Prediction_using_ML.ipynb  # Jupyter Notebook (EDA + experiments)
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Dataset
│
├── xgb_churn_model.pkl                # Trained XGBoost model
├── encoders.pkl                       # Saved label encoders
│
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation

🚀 Future Improvements

Use One-Hot Encoding / SMOTENC for better categorical handling
Perform hyperparameter tuning
Add cross-validation
Deploy the model using Flask or FastAPI
Build a web-based churn prediction system
Enable real-time predictions

👨‍💻 Author

DEEPAK BHATT
Computer Science Undergraduate
Interests: Software Development, Machine Learning, Data Science

📎 Disclaimer
This project is intended solely for educational and learning purposes and uses a publicly available dataset.