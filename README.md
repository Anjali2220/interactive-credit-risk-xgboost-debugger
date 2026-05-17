# 💳 Interactive Credit Risk XGBoost Debugger with Explainable AI

A complete end-to-end Explainable AI (XAI) based credit risk prediction system built using XGBoost, SHAP, and Streamlit.

This project combines machine learning prediction with explainability techniques to create a transparent loan approval system. The system predicts whether a loan application will be approved or rejected and explains the reasons behind the prediction using SHAP feature contributions.

---

## 📌 Project Overview

Loan approval systems are widely used in banks and financial institutions to evaluate applicants and make decisions regarding loan eligibility.

Traditional machine learning systems usually behave as "black-box" models where users receive only prediction results without understanding the reasoning behind the decision.

In this project, I developed an explainable loan prediction system that:

- Predicts loan approval status
- Explains model decisions
- Shows positive and negative factors affecting approval
- Detects model health issues
- Provides an interactive user dashboard

The project transforms a traditional prediction model into an interpretable decision-support system.

---

# 🚀 Objectives

The major goals of this project were:

- Build an accurate loan approval prediction system
- Implement Explainable AI techniques
- Understand feature influence on predictions
- Create an interactive Streamlit dashboard
- Perform model debugging and health analysis
- Improve transparency of machine learning decisions

---

# 🧠 Model Implemented

## XGBoost Classifier

XGBoost (Extreme Gradient Boosting) was selected because of its:

- High prediction performance
- Ability to handle structured tabular data
- Regularization capabilities
- Reduced overfitting
- Fast computation speed

### Training Configuration

```python
max_depth = 3
learning_rate = 0.05
n_estimators = 50
subsample = 0.8
reg_alpha = 1
reg_lambda = 1
scale_pos_weight = 2
random_state = 42
```

---

# 📂 Dataset Used

## Loan Prediction Dataset

The project uses a historical loan application dataset containing applicant information and loan approval status.

### Dataset Statistics

| Attribute | Value |
|---|---:|
| Dataset Rows | 614 |
| Original Features | 12 |
| Encoded Features | 14 |

---

### Features Used

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

---

# ⚙️ Data Preprocessing

The preprocessing pipeline included:

- Missing value handling
- Data cleaning
- Numerical feature imputation
- Categorical feature imputation
- One-Hot Encoding
- Feature transformation

### Missing Value Handling

#### Numerical Features

- Median Imputation

#### Categorical Features

- Mode Imputation

---

Purpose:

- Understand dataset characteristics
- Detect imbalance problems before model training

---

# ⚙️ System Architecture

```text
Loan Dataset
       ↓
Data Preprocessing
(Missing values + Encoding)
       ↓
Train-Test Split
       ↓
XGBoost Model Training
       ↓
Prediction System
       ↓
SHAP Explainability Module
       ↓
Model Debugging Module
       ↓
Streamlit Dashboard
       ↓
Final Loan Decision + Explanation
```

---

# 🔍 Explainable AI using SHAP

SHAP (Shapley Additive Explanations) was integrated to explain individual predictions.

SHAP helps answer questions like:

- Why was a loan approved?
- Why was a loan rejected?
- Which features contributed most?

---

## SHAP Interpretation Example

### Positive Factors Increasing Approval

| Feature | SHAP Contribution |
|---|---:|
| Credit History | +0.425 |
| Married | +0.095 |
| Applicant Income | +0.076 |
| Loan Amount | +0.073 |

---

### Negative Factors Reducing Approval

| Feature | SHAP Contribution |
|---|---:|
| Property Area | -0.164 |
| Coapplicant Income | -0.050 |
| Dependents | -0.006 |

---

### Interpretation

Example:

Credit_History → +0.425

This means:

- Credit history strongly pushed prediction toward approval
- Positive values increase approval probability
- Negative values reduce approval probability

General interpretation scale:

| SHAP Value | Influence Strength |
|---|---|
| ±0.02 | Very weak |
| ±0.10 | Moderate |
| ±0.40 | Strong |
| ±1.00 | Extremely strong |

---

# 📈 Model Performance

| Metric | Score |
|---|---:|
| Training Accuracy | 81.46%% |
| Testing Accuracy | 78.86%% |
| Precision | 75.96% |
| Recall | 98.75% |
| F1 Score | 85.87% |



---

# 🩺 Model Debugging Module

The debugging module was developed to analyze model reliability.

### Implemented checks

- Overfitting detection
- Train-test accuracy gap
- Missing value analysis
- Class imbalance analysis
- Feature correlation analysis
- Health report generation

---

# 🖥️ Streamlit Dashboard

An interactive Streamlit application was developed to:

- Visualize dataset information
- Display model metrics
- Generate predictions
- Explain decisions using SHAP
- Show model health reports

---

## Dashboard Features

✅ Dataset overview

✅ Model performance metrics

✅ SHAP explainability plots

✅ Individual prediction testing

✅ Approval explanation

✅ Model health analysis

---

# 📷 Dashboard Screenshots

## Dashboard Overview

<img src="screenshots/dashboard1.png" width="900">

---

## SHAP Explainability

<img src="screenshots/dashboard2.png" width="900">

---

## Loan Prediction Example

<img src="screenshots/dashboard3.png" width="900">

---

# 🛠️ Tech Stack

## Languages & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib

---
## ✅ What I Achieved

- Built an end-to-end credit risk prediction pipeline
- Integrated Explainable AI using SHAP
- Added real-time prediction capability
- Implemented model debugging system
- Developed an interactive Streamlit dashboard
- Performed complete validation testing

---

# 📚 What I Learned

Through this project I gained practical experience in:

## Machine Learning

- Classification models
- XGBoost optimization
- Feature engineering
- Model evaluation

## Explainable AI

- SHAP values
- Global explanations
- Local explanations
- Waterfall plots

## Machine Learning Engineering

- Data preprocessing pipelines
- Model debugging
- Error handling
- Validation testing

## Deployment

- Streamlit dashboard development
- Git and GitHub workflow

---

# 🔮 Future Improvements

- Add deep learning models
- Add cloud deployment
- Add MongoDB integration
- Add API-based prediction system
- Implement advanced explainability methods

---


GitHub:
https://github.com/Anjali2220/interactive-credit-risk-xgboost-debugger

---

