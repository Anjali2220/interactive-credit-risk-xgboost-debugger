import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/train_u6lujuX_CVtuZ9i.csv.xls")

# First 5 rows
print("\n First 5 Rows:")
print(df.head())

# Column names
print("\n Dataset Columns:")
print(df.columns)

# Dataset shape
print("\n Dataset Shape:")
print(df.shape)

# Missing values
print("\n Missing Values:")
print(df.isnull().sum())

# Data types
print("\n Data Types:")
print(df.dtypes)

# Target distribution
print("\n Loan Status Distribution:")
print(df["Loan_Status"].value_counts())


# Histogram for Applicant Income
plt.figure(figsize=(8,5))
sns.histplot(df["ApplicantIncome"], bins=30)

plt.title("Applicant Income Distribution")
plt.show()

# Loan status count plot
plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status", data=df)

plt.title("Loan Approval Distribution")
plt.show()

# Correlation heatmap

# Select numerical columns only
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()