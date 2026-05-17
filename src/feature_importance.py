import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("../data/train_u6lujuX_CVtuZ9i.csv.xls")

# Drop Loan_ID
df = df.drop("Loan_ID", axis=1)

# Numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Categorical columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

# Fill numerical missing values
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
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

# Train model
model = XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

print("\n Model Trained Successfully!")

# Extract feature importance

importance_scores = model.feature_importances_

print("\n Feature Importance Scores:")
print(importance_scores)

# Create feature importance dataframe

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance_scores
})

print("\n Feature Importance DataFrame:")
print(feature_importance_df)

# Sort feature importance

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n Ranked Feature Importance:")
print(feature_importance_df)


# Plot feature importance

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance_df.head(10)
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()

# Top 5 important features

print("\n Top 5 Important Features:")

top_features = feature_importance_df.head(5)

for index, row in top_features.iterrows():
    print(f"{row['Feature']} → Importance Score: {row['Importance']:.4f}")