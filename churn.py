# ==========================================
# Customer Churn Prediction - Final Script
# ==========================================

# --------- Import Libraries ---------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import pickle


# --------- Load Dataset ---------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

pd.set_option("display.max_columns", None)


# --------- Initial Inspection ---------
print(df.info())
print(df["Churn"].value_counts())


# --------- Drop Unnecessary Column ---------
df.drop(columns=["customerID"], inplace=True)


# --------- Fix TotalCharges Column ---------
# Replace blank spaces with 0 and convert to float
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0")
df["TotalCharges"] = df["TotalCharges"].astype(float)


# --------- Encode Categorical Variables ---------
# SMOTE requires all features to be numeric
categorical_columns = df.select_dtypes(include="object").columns

le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])


# --------- Final Data Check (IMPORTANT) ---------
# Ensure no object/string columns remain
print("\nData types after encoding:\n", df.dtypes)


# --------- Split Features and Target ---------
X = df.drop("Churn", axis=1)
y = df["Churn"]


# --------- Handle Class Imbalance with SMOTE ---------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# --------- Train-Test Split ---------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)


# ==================================
# Model 1: Decision Tree
# ==================================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# ==================================
# Model 2: Random Forest
# ==================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# ==================================
# Model 3: XGBoost (Final Model)
# ==================================
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost Results")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# --------- Save Final Model ---------
with open("xgb_churn_model.pkl", "wb") as file:
    pickle.dump(xgb, file)

print("\n✅ Model saved successfully as xgb_churn_model.pkl")
