import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

if "profile_saved" not in st.session_state:
    st.session_state.profile_saved = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Loan Risk Model",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Loan Default Risk Model")
st.caption("Decision-support system for retail loan risk assessment")

THRESHOLD = 0.3

# ---------------- SIDEBAR ----------------
st.sidebar.title("👤 Applicant Details")

# Optional image (use URL if you don't want local file)
st.sidebar.image("images/bank_image.png", use_container_width=True)


first_name = st.sidebar.text_input("First Name")
last_name = st.sidebar.text_input("Last Name")
job_title = st.sidebar.text_input("Job Title")
married = st.sidebar.radio("Marital Status", ["Single", "Married"])

if st.sidebar.button("💾 Save Changes"):
    if first_name.strip() and last_name.strip() and job_title.strip():
        st.session_state.profile_saved = True
        st.sidebar.success("Details saved successfully")
    else:
        st.sidebar.error("Please fill all required fields")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'datasets', 'bank_loan_data.csv')
    return pd.read_csv(data_path)

df = load_data()


# ---------------- MODEL PIPELINE ----------------
X = df.drop("default", axis=1)
y = df["default"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

for col in ["income", "loan_amount"]:
    X_train[col] = np.log1p(X_train[col])
    X_val[col] = np.log1p(X_val[col])
    X_test[col] = np.log1p(X_test[col])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------- MODEL EVALUATION ----------------
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_proba > THRESHOLD).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

# ---------------- DATA PREVIEW ----------------
st.subheader("🧾 Dataset Preview")
st.dataframe(df, use_container_width=True)

st.subheader("📊 Confusion Matrix")
st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual Good", "Actual Bad"],
        columns=["Predicted Good", "Predicted Bad"]
    ),
    use_container_width=True
)

st.info(
    f"""
**Business Interpretation**
- False Negatives (Bad loans approved): **{fn}**
- Threshold used: **{THRESHOLD}**
- Model prioritizes bank safety over approval rate
"""
)

# ---------------- GATED CUSTOMER CHECK ----------------
if not st.session_state.profile_saved:
    st.warning("⚠️ Complete applicant details in the sidebar to evaluate loan risk.")
else:
    st.subheader("🔍 Check for New Customer")

    st.markdown(f"""
    **Applicant Summary**
    - Name: {first_name} {last_name}
    - Job Title: {job_title}
    - Marital Status: {married}
    """)

    age = st.slider("Age", 21, 65, 25)
    income = st.number_input("Annual Income", 20000, 150000, 30000)
    credit_score = st.slider("Credit Score", 300, 850, 400)
    loan_amt = st.number_input("Loan Amount", 5000, 500000, 100000)
    existing_loans = st.slider("Existing Loans", 0, 5, 2)

    if st.button("🔮 Evaluate Customer"):
        user_input = pd.DataFrame({
            "age": [age],
            "income": [np.log1p(income)],
            "credit_score": [credit_score],
            "loan_amount": [np.log1p(loan_amt)],
            "existing_loans": [existing_loans]
        })

        user_input_scaled = scaler.transform(user_input)
        user_probability = model.predict_proba(user_input_scaled)[0, 1]
        prediction = int(user_probability > THRESHOLD)

        st.metric("Probability of Default", f"{user_probability:.2%}")

        if prediction == 1:
            st.error("❌ Loan Rejected — High Risk Applicant")
        else:
            st.success("✅ Loan Approved — Acceptable Risk")
