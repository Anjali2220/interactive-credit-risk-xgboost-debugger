import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("../data/train_u6lujuX_CVtuZ9i.csv.xls")

# Drop Loan_ID column
df = df.drop("Loan_ID", axis=1)

# Numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Categorical columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

# Fill numerical missing values with median
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop("Loan_Status_Y", axis=1)

y = df_encoded["Loan_Status_Y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = XGBClassifier()

# Train model
model.fit(X_train, y_train)

print("\n Model Trained Successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\n Predictions Generated!")


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# F1-score
f1 = f1_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n Confusion Matrix:")
print(cm)


# Plot confusion matrix

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# Classification report

print("\n Classification Report:")
print(classification_report(y_test, y_pred))


# Tuned model

tuned_model = XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    random_state=42
)

# Train tuned model
tuned_model.fit(X_train, y_train)

# Predictions
tuned_pred = tuned_model.predict(X_test)

# Tuned model accuracy

tuned_accuracy = accuracy_score(y_test, tuned_pred)

print("\n Tuned Model Accuracy:")
print(f"{tuned_accuracy:.4f}")