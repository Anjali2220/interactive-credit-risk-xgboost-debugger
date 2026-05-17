from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
# Load dataset
df = pd.read_csv("../data/train_u6lujuX_CVtuZ9i.csv.xls")

# Drop Loan_ID column
df = df.drop("Loan_ID", axis=1)

print("\n Dataset Loaded Successfully!")
print(df.head())

# Numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Categorical columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

print("\n Numerical Columns:")
print(numerical_cols)

print("\n Categorical Columns:")
print(categorical_cols)

# Fill missing values in numerical columns using median

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing values in categorical columns using mode

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check missing values again

print("\n Missing Values After Preprocessing:")
print(df.isnull().sum())

# One-hot encode categorical columns

df_encoded = pd.get_dummies(df, drop_first=True)

print("\n Encoded Dataset:")
print(df_encoded.head())

# Features and target

X = df_encoded.drop("Loan_Status_Y", axis=1)

y = df_encoded["Loan_Status_Y"]

print("\n Features Shape:")
print(X.shape)

print("\n Target Shape:")
print(y.shape)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n Training Features Shape:")
print(X_train.shape)

print("\n Testing Features Shape:")
print(X_test.shape)

print("\n Training Target Shape:")
print(y_train.shape)

print("\n Testing Target Shape:")
print(y_test.shape)

# Final verification

print("\n Final Processed Dataset Info:")
print(X_train.head())

print("\n No Missing Values Remaining:")
print(X_train.isnull().sum().sum())