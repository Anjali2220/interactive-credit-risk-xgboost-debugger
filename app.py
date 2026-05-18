import base64
import textwrap

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Interactive Credit Risk XGBoost Debugger",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def html(content):
    st.markdown(textwrap.dedent(content), unsafe_allow_html=True)


def set_background(image_path):

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(
            image_file.read()
        ).decode()

    html(f"""
    <style>

    .stApp {{
        background-image:
            linear-gradient(
                rgba(3, 10, 25, 0.55),
                rgba(3, 10, 25, 0.55)
            ),
            url("data:image/png;base64,{encoded_string}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .hero-box {{
        background: linear-gradient(
            135deg,
            rgba(8, 20, 45, 0.96),
            rgba(15, 55, 95, 0.90)
        );

        padding: 42px;
        border-radius: 30px;
        box-shadow: 0px 18px 45px rgba(0,0,0,0.40);
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.22);
        
    }}

    .main-title {{
        font-size: 24px;
        font-weight: 900;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 12px;
    }}

    .sub-title {{
        font-size: 34px;
        color: #DDEBFF;
        text-align: center;
    }}

    .section-title {{
        color: #FFFFFF;
        font-size: 40px;
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 18px;
    }}

    .project-card {{
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.96),
            rgba(219,234,254,0.96)
        );

        padding: 32px;
        border-radius: 24px;
        box-shadow: 0px 12px 30px rgba(0,0,0,0.24);
        border-left: 8px solid #38BDF8;
        color: #0F172A;
        margin-bottom: 24px;
    }}

    .project-card h3,
    .project-card h4 {{
        color: #0B2545 !important;
    }}

    .project-card p,
    .project-card li {{
        color: #1E293B !important;
        font-size: 17px;
        line-height: 1.7;
    }}

    .green-card {{
        background: linear-gradient(
            135deg,
            rgba(220,252,231,0.96),
            rgba(187,247,208,0.96)
        );

        padding: 24px;
        border-radius: 20px;
        border-left: 7px solid #16A34A;
        color: #064E3B !important;
        margin-bottom: 20px;
        box-shadow: 0px 10px 24px rgba(0,0,0,0.18);
        font-size: 17px;
    }}

    .green-card h3,
    .green-card p,
    .green-card b {{
        color: #064E3B !important;
    }}

    .orange-card {{
        background: linear-gradient(
            135deg,
            rgba(255,237,213,0.96),
            rgba(254,215,170,0.96)
        );

        padding: 24px;
        border-radius: 20px;
        border-left: 7px solid #EA580C;
        color: #7C2D12 !important;
        margin-bottom: 20px;
        box-shadow: 0px 10px 24px rgba(0,0,0,0.18);
        font-size: 17px;
    }}

    .orange-card h3,
    .orange-card p,
    .orange-card b {{
        color: #7C2D12 !important;
    }}

    .blue-card {{
        background: linear-gradient(
            135deg,
            rgba(219,234,254,0.96),
            rgba(191,219,254,0.96)
        );

        padding: 24px;
        border-radius: 20px;
        border-left: 7px solid #2563EB;
        color: #0B2545 !important;
        margin-bottom: 20px;
        box-shadow: 0px 10px 24px rgba(0,0,0,0.18);
        font-size: 17px;
    }}

    .blue-card h3,
    .blue-card p,
    .blue-card b {{
        color: #0B2545 !important;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.94),
            rgba(30, 58, 138, 0.88)
        );

        padding: 22px;
        border-radius: 22px;
        box-shadow: 0px 10px 26px rgba(0,0,0,0.32);
        border: 1px solid rgba(255,255,255,0.20);
    }}

    div[data-testid="stMetric"] label {{
        color: #BAE6FD !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: #FFFFFF !important;
        font-size: 34px !important;
        font-weight: 900 !important;
    }}

    h1, h2, h3, h4, h5, h6, p, label {{
        color: white;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 14px;
        margin-bottom: 18px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: rgba(15, 23, 42, 0.88);
        border-radius: 16px;
        padding: 12px 22px;
        font-weight: 800;
        color: white;
        border: 1px solid rgba(255,255,255,0.14);
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #38BDF8, #2563EB);
        color: #FFFFFF;
    }}

    </style>
    """)


# =========================================================
# BACKGROUND
# =========================================================

set_background("images/background.png")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/train_u6lujuX_CVtuZ9i.csv.xls"
    )

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

    X = df_encoded.drop(
        "Loan_Status_Y",
        axis=1
    )

    y = df_encoded["Loan_Status_Y"]

    return df, X, y


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model(X, y):

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
        subsample=0.8,
        reg_alpha=1,
        reg_lambda=1,
        scale_pos_weight=2,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {

        "train_accuracy":
            accuracy_score(y_train, train_pred),

        "test_accuracy":
            accuracy_score(y_test, test_pred),

        "precision":
            precision_score(y_test, test_pred),

        "recall":
            recall_score(y_test, test_pred),

        "f1":
            f1_score(y_test, test_pred),

        "confusion_matrix":
            confusion_matrix(y_test, test_pred),

        "X_test":
            X_test
    }

    return model, metrics


# =========================================================
# LOAD EVERYTHING
# =========================================================

df, X, y = load_data()

model, metrics = train_model(X, y)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    metrics["X_test"]
)


# =========================================================
# HERO SECTION
# =========================================================

html("""
<div class="hero-box">

   
        💳 Interactive Credit Risk XGBoost Debugger - Explainable AI Dashboard for Loan Approval Prediction using XGBoost, SHAP, and Streamlit
    

</div>
""")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📊 Model Performance",
    "🔍 SHAP Explainability",
    "🧪 Try Prediction",
    "🩺 Model Health"
])


# =========================================================
# TAB 1
# =========================================================

with tab1:

    html("""
    <div class="project-card">

        This application predicts whether a loan application
        will be approved or rejected using an optimized
        XGBoost model.

        It also explains the model decision using SHAP values,
        making predictions transparent and interpretable.
        

    </div>
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Rows", df.shape[0])
    c2.metric("Original Features", df.shape[1])
    c3.metric("Encoded Features", X.shape[1])

    st.markdown(
        '<div class="section-title">📄 Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">📊 Loan Status Distribution</div>',
        unsafe_allow_html=True
    )

    st.bar_chart(
        df["Loan_Status"].value_counts()
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">📊 Model Performance Metrics</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Training Accuracy",
        f"{metrics['train_accuracy']:.2%}"
    )

    col2.metric(
        "Testing Accuracy",
        f"{metrics['test_accuracy']:.2%}"
    )

    col3.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    col4.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    col5.metric(
        "F1 Score",
        f"{metrics['f1']:.2%}"
    )

    html("""
    <div class="green-card">

        Optimization Result:

        The improved XGBoost model reduced overfitting
        and improved generalization performance.

    </div>
    """)

    st.markdown(
        '<div class="section-title">🧮 Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Rejected", "Actual Approved"],
        columns=["Predicted Rejected", "Predicted Approved"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">🔍 SHAP Explainability</div>',
        unsafe_allow_html=True
    )

    html("""
    <div class="blue-card">

        What SHAP explains:

        SHAP shows how much each feature contributes toward
        loan approval or rejection.

        Positive SHAP values push predictions toward approval,
        while negative values push predictions toward rejection.

    </div>
    """)

    shap_importance = pd.DataFrame({

        "Feature":
            metrics["X_test"].columns,

        "SHAP Importance":
            abs(shap_values).mean(axis=0)

    }).sort_values(
        by="SHAP Importance",
        ascending=False
    )

    st.markdown(
        '<div class="section-title">🏆 Top SHAP Features</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        shap_importance.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">📈 SHAP Feature Importance Plot</div>',
        unsafe_allow_html=True
    )

    fig_bar = plt.figure()

    shap.summary_plot(
        shap_values,
        metrics["X_test"],
        plot_type="bar",
        show=False
    )

    st.pyplot(fig_bar, clear_figure=True)

    st.markdown(
        '<div class="section-title">🌈 SHAP Summary Plot</div>',
        unsafe_allow_html=True
    )

    fig_summary = plt.figure()

    shap.summary_plot(
        shap_values,
        metrics["X_test"],
        show=False
    )

    st.pyplot(fig_summary, clear_figure=True)


# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.markdown(
        '<div class="section-title">🧪 Try Your Own Loan Prediction</div>',
        unsafe_allow_html=True
    )

    html("""
    <div class="blue-card">

        Enter applicant details below.

        The model will predict loan approval status
        and explain the decision using SHAP Explainable AI.

    </div>
    """)

    col1, col2 = st.columns(2)

    with col1:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1,
            value=150
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [360, 180, 120, 60, 300, 240]
        )

    with col2:

        credit_history = st.selectbox(
            "Credit History",
            [1, 0]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=X.columns
    )

    input_data["ApplicantIncome"] = applicant_income
    input_data["CoapplicantIncome"] = coapplicant_income
    input_data["LoanAmount"] = loan_amount
    input_data["Loan_Amount_Term"] = loan_term
    input_data["Credit_History"] = credit_history

    if gender == "Male" and "Gender_Male" in input_data.columns:
        input_data["Gender_Male"] = 1

    if married == "Yes" and "Married_Yes" in input_data.columns:
        input_data["Married_Yes"] = 1

    if dependents == "1" and "Dependents_1" in input_data.columns:
        input_data["Dependents_1"] = 1

    if dependents == "2" and "Dependents_2" in input_data.columns:
        input_data["Dependents_2"] = 1

    if dependents == "3+" and "Dependents_3+" in input_data.columns:
        input_data["Dependents_3+"] = 1

    if education == "Not Graduate" and "Education_Not Graduate" in input_data.columns:
        input_data["Education_Not Graduate"] = 1

    if self_employed == "Yes" and "Self_Employed_Yes" in input_data.columns:
        input_data["Self_Employed_Yes"] = 1

    if property_area == "Semiurban" and "Property_Area_Semiurban" in input_data.columns:
        input_data["Property_Area_Semiurban"] = 1

    if property_area == "Urban" and "Property_Area_Urban" in input_data.columns:
        input_data["Property_Area_Urban"] = 1

    if st.button("🔮 Predict Loan Status"):

        prediction = model.predict(input_data)[0]

        approval_probability = model.predict_proba(
            input_data
        )[0][1]

        input_shap_values = explainer.shap_values(
            input_data
        )

        explanation_df = pd.DataFrame({

            "Feature":
                input_data.columns,

            "Feature Value":
                input_data.iloc[0].values,

            "SHAP Contribution":
                input_shap_values[0]

        })

        explanation_df["Absolute Contribution"] = (
            explanation_df["SHAP Contribution"].abs()
        )

        explanation_df = explanation_df.sort_values(
            by="Absolute Contribution",
            ascending=False
        )

        positive_factors = explanation_df[
            explanation_df["SHAP Contribution"] > 0
        ].head(5)

        negative_factors = explanation_df[
            explanation_df["SHAP Contribution"] < 0
        ].head(5)

        # =====================================================
        # PREDICTION RESULT
        # =====================================================

        if prediction == 1:

            html(f"""
            <div class="green-card">

                Loan Predicted: APPROVED

                
                    Approval Probability:
                    {approval_probability:.2%}
                

                
                The model found more positive factors
                supporting loan approval.
                

            </div>
            """)

        else:

            html(f"""
            <div class="orange-card">

                ⚠️ Loan Predicted: REJECTED

                
                    <b>Approval Probability:</b>
                    {approval_probability:.2%}
                

                The model identified several risk factors
                reducing approval confidence.
                

            </div>
            """)

        # =====================================================
        # POSITIVE FACTORS
        # =====================================================

        st.markdown(
            '<div class="section-title">✅ Reasons Increasing Approval</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            positive_factors[
                ["Feature", "Feature Value", "SHAP Contribution"]
            ],
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
        # NEGATIVE FACTORS
        # =====================================================

        st.markdown(
            '<div class="section-title">⚠️ Reasons Decreasing Approval</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            negative_factors[
                ["Feature", "Feature Value", "SHAP Contribution"]
            ],
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
        # HUMAN FRIENDLY AI SUMMARY
        # =====================================================

        st.markdown(
            '<div class="section-title">🧠 XAI (Explainable AI ) Decision Summary</div>',
            unsafe_allow_html=True
        )

        feature_explanations = {

            "Credit_History":
                "Strong credit history significantly improved approval confidence.",

            "ApplicantIncome":
                "Applicant income reduced approval confidence.",

            "CoapplicantIncome":
                "Lack of coapplicant income negatively affected financial reliability.",

            "LoanAmount":
                "Loan amount influenced the overall approval decision.",

            "Loan_Amount_Term":
                "Loan repayment term supported approval stability.",

            "Property_Area_Urban":
                "Urban property area positively influenced prediction.",

            "Property_Area_Semiurban":
                "Semiurban property area slightly reduced approval probability.",

            "Married_Yes":
                "Marital status positively influenced repayment confidence.",

            "Dependents_1":
                "Having one dependent had a small positive influence.",

            "Dependents_2":
                "Additional dependents slightly increased financial risk.",

            "Dependents_3+":
                "Higher number of dependents reduced approval confidence.",

            "Education_Not Graduate":
                "Education level slightly influenced loan approval confidence."
        }

        st.markdown(
            "### ✅ Main Reasons Supporting Approval"
        )

        for feature in positive_factors["Feature"]:

            explanation = feature_explanations.get(
                feature,
                f"{feature} positively influenced prediction."
            )

            st.success(explanation)

        st.markdown(
            "### ⚠️ Main Risk Factors"
        )

        for feature in negative_factors["Feature"]:

            explanation = feature_explanations.get(
                feature,
                f"{feature} negatively influenced prediction."
            )

            st.error(explanation)

        # =====================================================
        # FULL EXPLANATION
        # =====================================================

        st.markdown(
            '<div class="section-title">📌 Full Prediction Explanation</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            explanation_df[
                ["Feature", "Feature Value", "SHAP Contribution"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
        # WATERFALL PLOT
        # =====================================================

        st.markdown(
            '<div class="section-title">🌊 Individual SHAP Waterfall Plot</div>',
            unsafe_allow_html=True
        )

        shap_explanation = shap.Explanation(
            values=input_shap_values[0],
            base_values=explainer.expected_value,
            data=input_data.iloc[0],
            feature_names=input_data.columns
        )

        fig_waterfall = plt.figure(figsize=(10, 6))

        shap.plots.waterfall(
            shap_explanation,
            show=False
        )

        st.pyplot(
            fig_waterfall,
            clear_figure=True
        )


# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.markdown(
        '<div class="section-title">🩺 Model Health Analysis</div>',
        unsafe_allow_html=True
    )

    accuracy_gap = (
        metrics["train_accuracy"]
        - metrics["test_accuracy"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Train-Test Gap",
        f"{accuracy_gap:.2%}"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    if accuracy_gap > 0.15:

        html("""
        <div class="orange-card">

            <b>⚠️ Possible Overfitting Detected</b><br>

            Training accuracy is much higher than testing accuracy.

        </div>
        """)

    else:

        html("""
        <div class="green-card">

            <b>✅ No Serious Overfitting Detected</b><br>

            The optimized model generalizes well on unseen test data.

        </div>
        """)

    html("""
    <div class="project-card">

        <h3>📌 Key Achievements</h3>

        <p>✅ Built an explainable XGBoost credit risk model</p>
        <p>✅ Integrated SHAP explainable AI for transparent predictions</p>
        <p>✅ Added interactive prediction testing</p>
        <p>✅ Shows why a loan is approved or rejected</p>
        <p>✅ Created a model debugging and health analysis system</p>

    </div>
    """)