import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("../data/train_u6lujuX_CVtuZ9i.csv.xls")

# Drop Loan_ID because it is only an identifier
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

# Convert True/False to 1/0
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
# 6. Train XGBoost Model
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

print("\nModel Trained Successfully!")


# ==============================
# 7. Create SHAP Explainer
# ==============================

explainer = shap.TreeExplainer(model)

print("\nSHAP Explainer Created!")


# ==============================
# 8. Generate SHAP Values
# ==============================

shap_values = explainer.shap_values(X_test)

print("\nSHAP Values Generated!")


# ==============================
# 9. SHAP Feature Importance Scores
# ==============================

shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "SHAP Importance": abs(shap_values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="SHAP Importance",
    ascending=False
)

print("\nSHAP Feature Importance Ranking:")
print(shap_importance)

print("\nTop 5 Most Influential Features:")

top_5 = shap_importance.head(5)

for index, row in top_5.iterrows():
    print(f"{row['Feature']} → SHAP Score: {row['SHAP Importance']:.4f}")


# ==============================
# 10. SHAP Summary Plot
# ==============================

print("\nOpening SHAP Summary Plot...")

shap.summary_plot(
    shap_values,
    X_test
)


# ==============================
# 11. SHAP Bar Plot
# ==============================

print("\nOpening SHAP Bar Plot...")

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)


# ==============================
# 12. Individual Prediction Explanation
# ==============================

sample_index = 0

print("\nExplaining Prediction for Sample:", sample_index)

single_prediction = model.predict(X_test.iloc[[sample_index]])[0]
single_probability = model.predict_proba(X_test.iloc[[sample_index]])[0][1]

print("\nPrediction Result:")
if single_prediction == 1:
    print("Predicted Loan Status: Approved")
else:
    print("Predicted Loan Status: Rejected")

print(f"Approval Probability: {single_probability:.4f}")


# ==============================
# 13. Individual SHAP Contributions
# ==============================

sample_shap_values = shap_values[sample_index]

sample_features = pd.DataFrame({
    "Feature": X_test.columns,
    "Feature Value": X_test.iloc[sample_index].values,
    "SHAP Contribution": sample_shap_values
})

sample_features["Absolute Contribution"] = sample_features["SHAP Contribution"].abs()

sample_features = sample_features.sort_values(
    by="Absolute Contribution",
    ascending=False
)

print("\nIndividual Prediction Explanation:")
print(sample_features[["Feature", "Feature Value", "SHAP Contribution"]].head(10))


# ==============================
# 14. Human-Readable Explanation
# ==============================

positive_factors = sample_features[sample_features["SHAP Contribution"] > 0].head(5)
negative_factors = sample_features[sample_features["SHAP Contribution"] < 0].head(5)

print("\nHuman-Readable Explanation:")

print("\nPositive Factors Increasing Approval Probability:")
for index, row in positive_factors.iterrows():
    print(
        f"{row['Feature']} = {row['Feature Value']} "
        f"increased approval probability by {row['SHAP Contribution']:.4f}"
    )

print("\nNegative Factors Decreasing Approval Probability:")
for index, row in negative_factors.iterrows():
    print(
        f"{row['Feature']} = {row['Feature Value']} "
        f"decreased approval probability by {row['SHAP Contribution']:.4f}"
    )


# ==============================
# 15. SHAP Waterfall Plot
# ==============================

print("\nOpening SHAP Waterfall Plot...")

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_index],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_index],
        feature_names=X_test.columns
    )
)

plt.show()