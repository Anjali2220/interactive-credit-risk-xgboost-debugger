import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("../data/train_u6lujuX_CVtuZ9i.csv.xls")
df = df.drop("Loan_ID", axis=1)


# ==============================
# 2. Handle Missing Values
# ==============================

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ==============================
# 3. Encode Categorical Columns
# ==============================

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.astype(int)


# ==============================
# 4. Split Features and Target
# ==============================

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
# 6. Baseline Model
# ==============================

baseline_model = XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    random_state=42,
    eval_metric="logloss"
)

baseline_model.fit(X_train, y_train)

baseline_train_pred = baseline_model.predict(X_train)
baseline_test_pred = baseline_model.predict(X_test)


# ==============================
# 7. Improved Model
# ==============================

improved_model = XGBClassifier(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=50,
    subsample=0.8,
    reg_alpha=1,
    reg_lambda=1,
    scale_pos_weight=2,
    random_state=42,
    eval_metric="logloss"
)

improved_model.fit(X_train, y_train)

improved_train_pred = improved_model.predict(X_train)
improved_test_pred = improved_model.predict(X_test)


# ==============================
# 8. Evaluation Function
# ==============================

def evaluate_model(model_name, y_train, train_pred, y_test, test_pred):
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    accuracy_gap = train_accuracy - test_accuracy

    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    return {
        "Model": model_name,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy,
        "Accuracy Gap": accuracy_gap,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


baseline_results = evaluate_model(
    "Baseline XGBoost",
    y_train,
    baseline_train_pred,
    y_test,
    baseline_test_pred
)

improved_results = evaluate_model(
    "Improved XGBoost",
    y_train,
    improved_train_pred,
    y_test,
    improved_test_pred
)


# ==============================
# 9. Compare Results
# ==============================

comparison_df = pd.DataFrame([baseline_results, improved_results])

print("\nModel Comparison:")
print(comparison_df)


# ==============================
# 10. Detailed Reports
# ==============================

print("\nBaseline Model Confusion Matrix:")
print(confusion_matrix(y_test, baseline_test_pred))

print("\nBaseline Model Classification Report:")
print(classification_report(y_test, baseline_test_pred))


print("\nImproved Model Confusion Matrix:")
print(confusion_matrix(y_test, improved_test_pred))

print("\nImproved Model Classification Report:")
print(classification_report(y_test, improved_test_pred))


# ==============================
# 11. Save Comparison Report
# ==============================

comparison_df.to_csv("../model_optimization_comparison.csv", index=False)

print("\nModel optimization comparison saved successfully!")
print("File saved as: model_optimization_comparison.csv")


# ==============================
# 12. Interpretation
# ==============================

baseline_gap = baseline_results["Accuracy Gap"]
improved_gap = improved_results["Accuracy Gap"]

print("\nInterpretation:")

if improved_gap < baseline_gap:
    print("Improved model reduced overfitting compared to the baseline model.")
else:
    print("Improved model did not reduce overfitting. Further tuning may be required.")

if improved_results["F1-Score"] >= baseline_results["F1-Score"]:
    print("Improved model maintained or improved F1-score.")
else:
    print("Improved model reduced F1-score slightly. This may still be acceptable if overfitting reduced.")

print("\nStage 9 completed successfully!")