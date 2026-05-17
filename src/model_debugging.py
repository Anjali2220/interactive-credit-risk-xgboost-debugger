import os
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


# ==============================
# 1. Load Dataset
# ==============================

df_original = pd.read_csv("../data/train_u6lujuX_CVtuZ9i.csv.xls")

df = df_original.copy()

# Drop ID column
df = df.drop("Loan_ID", axis=1)


# ==============================
# 2. Missing Value Analysis BEFORE preprocessing
# ==============================

missing_percent = (df.isnull().sum() / len(df)) * 100
high_missing = missing_percent[missing_percent > 20]


# ==============================
# 3. Handle Missing Values
# ==============================

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ==============================
# 4. Encode Data
# ==============================

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.astype(int)

X = df_encoded.drop("Loan_Status_Y", axis=1)
y = df_encoded["Loan_Status_Y"]


# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# 6. Train Model
# ==============================

model = XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)


# ==============================
# 7. Detect Overfitting
# ==============================

accuracy_gap = train_accuracy - test_accuracy

if accuracy_gap > 0.15:
    overfitting_status = "Possible overfitting detected"
    overfitting_suggestion = "Training accuracy is much higher than testing accuracy. Try reducing max_depth, using regularization, or collecting more data."
else:
    overfitting_status = "No serious overfitting detected"
    overfitting_suggestion = "Training and testing accuracy are reasonably close."


# ==============================
# 8. Detect Class Imbalance
# ==============================

class_counts = y.value_counts()
class_percent = y.value_counts(normalize=True) * 100

majority_class_percent = class_percent.max()
minority_class_percent = class_percent.min()

if majority_class_percent > 70:
    imbalance_status = "Class imbalance detected"
    imbalance_suggestion = "One class dominates the dataset. Consider using class weights, resampling, or evaluation metrics like F1-score and recall."
else:
    imbalance_status = "No severe class imbalance detected"
    imbalance_suggestion = "Class distribution is acceptable, but still monitor precision and recall."


# ==============================
# 9. Detect Highly Correlated Features
# ==============================

corr_matrix = X.corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []

for column in upper_triangle.columns:
    correlated_features = upper_triangle[column][upper_triangle[column] > 0.85]
    for index, value in correlated_features.items():
        high_corr_pairs.append((index, column, value))

if len(high_corr_pairs) > 0:
    correlation_status = "Highly correlated features detected"
    correlation_suggestion = "Some features are strongly correlated. Consider removing one feature from each highly correlated pair."
else:
    correlation_status = "No highly correlated features detected"
    correlation_suggestion = "No strong multicollinearity issue found."


# ==============================
# 10. Detect Excessive Missing Values
# ==============================

if len(high_missing) > 0:
    missing_status = "Excessive missing values detected"
    missing_suggestion = "Some columns have more than 20% missing values. Consider removing or carefully imputing these columns."
else:
    missing_status = "No excessive missing values detected"
    missing_suggestion = "Missing values are manageable and were handled using median/mode imputation."


# ==============================
# 11. Feature Leakage Warning
# ==============================

leakage_warning = "No direct feature leakage detected based on column names."

suspicious_keywords = ["status", "approved", "approval", "target", "result"]

for col in X.columns:
    if any(keyword in col.lower() for keyword in suspicious_keywords):
        leakage_warning = f"Possible leakage warning: Feature '{col}' may be related to the target."
        break


# ==============================
# 12. Create Model Health Report
# ==============================

report = f"""
MODEL HEALTH REPORT
===================

Project:
Interactive Credit Risk XGBoost Debugger using SHAP and Streamlit

Report Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


1. DATASET OVERVIEW
-------------------
Total rows: {df.shape[0]}
Total columns after dropping Loan_ID: {df.shape[1]}
Total features after encoding: {X.shape[1]}
Target column: Loan_Status_Y


2. MODEL PERFORMANCE
--------------------
Training Accuracy: {train_accuracy:.4f}
Testing Accuracy: {test_accuracy:.4f}
Accuracy Gap: {accuracy_gap:.4f}

Overfitting Status:
{overfitting_status}

Suggestion:
{overfitting_suggestion}


3. CLASS IMBALANCE CHECK
------------------------
Class Counts:
{class_counts.to_string()}

Class Percentages:
{class_percent.round(2).to_string()}

Imbalance Status:
{imbalance_status}

Suggestion:
{imbalance_suggestion}


4. MISSING VALUE CHECK
----------------------
Missing Values Before Preprocessing (%):
{missing_percent.round(2).to_string()}

Missing Value Status:
{missing_status}

Suggestion:
{missing_suggestion}


5. HIGHLY CORRELATED FEATURES CHECK
-----------------------------------
Correlation Status:
{correlation_status}

Highly Correlated Feature Pairs:
"""

if high_corr_pairs:
    for f1, f2, corr in high_corr_pairs:
        report += f"\n{f1} and {f2}: correlation = {corr:.4f}"
else:
    report += "\nNo feature pairs above correlation threshold 0.85."

report += f"""


Suggestion:
{correlation_suggestion}


6. FEATURE LEAKAGE CHECK
------------------------
{leakage_warning}


7. CONFUSION MATRIX
-------------------
{confusion_matrix(y_test, test_pred)}


8. CLASSIFICATION REPORT
------------------------
{classification_report(y_test, test_pred)}


9. FINAL DEBUGGING SUMMARY
--------------------------
- {overfitting_status}
- {imbalance_status}
- {missing_status}
- {correlation_status}
- {leakage_warning}


10. FINAL RECOMMENDATIONS
-------------------------
1. Use SHAP explanations to understand important model decisions.
2. Monitor rejection class performance because the dataset has more approved loans than rejected loans.
3. Try class balancing if recall for rejected loans remains low.
4. Tune XGBoost hyperparameters further for better generalization.
5. Avoid using ID columns such as Loan_ID because they do not add useful predictive value.
"""


# ==============================
# 13. Save Report in Project Folder and Desktop
# ==============================

project_report_path = "../model_health_report.txt"

desktop_path = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "model_health_report.txt"
)

with open(project_report_path, "w") as file:
    file.write(report)

with open(desktop_path, "w") as file:
    file.write(report)


print("\nModel Debugging Completed Successfully!")
print("\nModel Health Report saved in project folder:")
print(project_report_path)

print("\nModel Health Report saved on Desktop:")
print(desktop_path)

print("\nSummary:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Overfitting Status: {overfitting_status}")
print(f"Class Imbalance Status: {imbalance_status}")
print(f"Missing Value Status: {missing_status}")
print(f"Correlation Status: {correlation_status}")