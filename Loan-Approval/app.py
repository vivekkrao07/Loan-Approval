# app.py
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scripts.preprocess import load_and_preprocess_data

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction App")

# -----------------------
# Sidebar: Load Dataset
# -----------------------
st.sidebar.header("Step 1: Load Dataset")
csv_path = st.sidebar.text_input("Enter CSV path:", "data/ProcessedLoan.csv")

try:
    X, y = load_and_preprocess_data(csv_path)

    # Drop raw LoanAmount (we use Loan_to_Income)
    if 'LoanAmount' in X.columns:
        X = X.drop(columns=['LoanAmount'])

    # Convert target to numeric
    y = y.map({'N': 0, 'Y': 1})

    st.sidebar.success("Dataset loaded successfully!")

    # -----------------------
    # Show Full Processed Table
    # -----------------------
    st.subheader("📋 Full Processed Data Table")
    st.dataframe(pd.concat([X, y], axis=1))

except Exception as e:
    st.sidebar.error(f"Error loading CSV: {e}")
    st.stop()

# -----------------------
# Train/Test Split & Model
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# -----------------------
# Display Model Metrics
# -----------------------
y_pred = model.predict(X_test)
st.subheader("📊 Model Performance on Test Set")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.write(f"**Precision:** {precision_score(y_test, y_pred, zero_division=0):.2f}")
st.write(f"**Recall:** {recall_score(y_test, y_pred, zero_division=0):.2f}")
st.write(f"**F1 Score:** {f1_score(y_test, y_pred, zero_division=0):.2f}")

# -----------------------
# Display Decision Tree Image
# -----------------------
st.subheader("🌳 Decision Tree Visualization")
try:
    tree_image = Image.open("tree.png")  # replace with your actual path
    st.image(tree_image, caption="Decision Tree for Loan Approval", use_container_width=True)
except Exception as e:
    st.error(f"Could not load decision tree image: {e}")

# -----------------------
# New Loan Prediction Form
# -----------------------
st.subheader("💡 Predict a New Loan Application")
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        married = st.selectbox("Married", ["Yes", "No"], index=1)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
        self_employed = st.selectbox("Self Employed", ["Yes", "No"], index=1)
    
    with col2:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0)
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=0)
        coapplicant_income = 0
        if married == "Yes":
            coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=0)
        loan_term = st.number_input("Loan Term (months)", min_value=0, value=0)
        credit_history = st.selectbox("Credit History", ["Good", "Bad"], index=0)

    submitted = st.form_submit_button("Predict Loan Status")

    if submitted:
        # Mapping dictionaries
        GENDER_MAP = {'Male':1,'Female':0}
        MARRIED_MAP = {'Yes':1,'No':0}
        DEPENDENTS_MAP = {'0':0,'1':1,'2':2,'3+':3}
        EDUCATION_MAP = {'Graduate':1,'Not Graduate':0}
        SELF_EMPLOYED_MAP = {'Yes':1,'No':0}
        PROPERTY_AREA_MAP = {'Urban':2,'Semiurban':1,'Rural':0}
        CREDIT_MAP = {'Good':1,'Bad':0}

        dep_numeric = DEPENDENTS_MAP[dependents]

        # Compute Loan-to-Income ratio and total income
        loan_to_income = loan_amount / (applicant_income + coapplicant_income + 1)
        total_income = applicant_income + coapplicant_income

        # Prepare DataFrame
        new_loan = pd.DataFrame([{
            'Gender': GENDER_MAP[gender],
            'Married': MARRIED_MAP[married],
            'Dependents': dep_numeric,
            'Education': EDUCATION_MAP[education],
            'Self_Employed': SELF_EMPLOYED_MAP[self_employed],
            'Property_Area': PROPERTY_AREA_MAP[property_area],
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'Loan_Amount_Term': loan_term,
            'Credit_History': CREDIT_MAP[credit_history],
            'Loan_to_Income': loan_to_income
        }])

        # Add missing columns
        for col in X_train.columns:
            if col not in new_loan.columns:
                new_loan[col] = 0

        # Reorder to match training columns
        new_loan = new_loan[X_train.columns]

        # Predict using model and hybrid rules
        try:
            prediction = model.predict(new_loan)

            reason_parts = []

            # Automatic approval if total income covers loan
            if total_income >= loan_amount:
                result = "Approved ✅"
                reason = "Reason: Total income sufficient to cover loan"
            else:
                # Hybrid rules applied if total income < loan
                if CREDIT_MAP[credit_history] == 0:
                    reason_parts.append("Bad Credit History")
                if loan_to_income > 0.5:
                    reason_parts.append("Loan too high compared to Income")
                if applicant_income < 30000 and loan_amount > 500000:
                    reason_parts.append("Low Income & High Loan")
                if dep_numeric >= 3 and applicant_income < 40000:
                    reason_parts.append("Too many dependents for income")
                if EDUCATION_MAP[education] == 0 and loan_to_income > 0.4:
                    reason_parts.append("Non-graduate with high loan-to-income ratio")

                if len(reason_parts) > 0:
                    result = "Not Approved ❌"
                    reason = "Reason: " + ", ".join(reason_parts)
                else:
                    result = "Approved ✅"
                    reason = "Reason: Meets all criteria"

            st.success(f"Prediction Result: {result}")
            st.info(reason)

        except Exception as e:
            st.error(f"Error predicting: {e}")
