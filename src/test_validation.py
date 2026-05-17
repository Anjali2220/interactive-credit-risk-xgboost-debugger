import os
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ==========================================
# DATA PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "train_u6lujuX_CVtuZ9i.csv.xls"
)


# ==========================================
# TEST 1
# ==========================================

def test_dataset_loading():

    df = pd.read_csv(DATA_PATH)

    assert not df.empty, "Dataset is empty"

    print("✅ Dataset loaded successfully")


# ==========================================
# TEST 2
# ==========================================

def test_required_columns():

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Loan_ID",
        "ApplicantIncome",
        "LoanAmount",
        "Credit_History",
        "Loan_Status"
    ]

    for col in required_columns:

        assert col in df.columns, f"Missing: {col}"

    print("✅ Required columns exist")


# ==========================================
# TEST 3
# ==========================================

def test_preprocessing():

    df = pd.read_csv(DATA_PATH)

    df = df.drop("Loan_ID", axis=1)

    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in numerical_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical_cols:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    df_encoded = pd.get_dummies(
        df,
        drop_first=True
    )

    df_encoded = df_encoded.astype(int)

    assert (
        df_encoded.isnull().sum().sum() == 0
    )

    print("✅ Preprocessing successful")


# ==========================================
# TEST 4
# ==========================================

def test_model_training():

    df = pd.read_csv(DATA_PATH)

    df = df.drop(
        "Loan_ID",
        axis=1
    )

    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in numerical_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical_cols:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    df_encoded = pd.get_dummies(
        df,
        drop_first=True
    )

    X = df_encoded.drop(
        "Loan_Status_Y",
        axis=1
    )

    y = df_encoded[
        "Loan_Status_Y"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=50,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test.head(1)
    )

    assert prediction[0] in [0,1]

    print("✅ Model training successful")

    return model, X_test


# ==========================================
# TEST 5
# ==========================================

def test_shap(model, X_test):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_test.head(5)
    )

    assert shap_values is not None

    print("✅ SHAP successful")


# ==========================================
# RUN ALL TESTS
# ==========================================

def run_all_tests():

    print("\nRunning Project Validation Tests...\n")

    test_dataset_loading()

    test_required_columns()

    test_preprocessing()

    model, X_test = test_model_training()

    test_shap(
        model,
        X_test
    )

    print("\n🎉 ALL TESTS PASSED")


if __name__ == "__main__":

    run_all_tests()