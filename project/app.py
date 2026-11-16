%%writefile app.py
import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Telecom Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dynamic configuration to allow local, docker, and cloud environments to connect correctly
def get_api_url():
    # Prioritize the environment variable set in Render's dashboard settings
    if "API_URL" in os.environ:
        return os.environ["API_URL"]
    # Fallback order: Check docker-compose network container 'api', else default to localhost
    try:
        requests.get("http://api:8000/", timeout=1)
        return "http://api:8000/predict"
    except Exception:
        return "http://127.0.0.1:8000/predict"

API_URL = get_api_url()

st.title("📊 Telecom Customer Churn Dashboard")
st.markdown(f"Adjust customer parameters on the left sidebar to communicate directly with the **FastAPI ML Server** at `{API_URL}`!")

# Sidebar input setup
st.sidebar.header("👤 Customer Profile")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * MonthlyCharges))

if st.sidebar.button("Run Predictor Inference", type="primary"):
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        with st.spinner("Fetching response from ML Service..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            churn_probability = result['churn_probability'] / 100.0

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{result['churn_probability']}%")
            with col2:
                if result['churn_prediction'] == 1:
                    st.error(f"🚨 Status: {result['status']}")
                else:
                    st.success(f"✅ Status: {result['status']}")

            # Strategic Key Takeaways
            st.subheader("ℹ️ Strategic Key Takeaways")
            if churn_probability >= 0.80:
                st.error("🚨 The Customer is at **Critical Churn Risk:** Immediate intervention requested. Trigger direct discount vouchers or escalate manually to premium customer support.")
            elif 0.65 <= churn_probability < 0.80:
                st.warning("⚠️ The Customer is at **High Churn Risk:** Please offer long-term contract migrations or value-added services like Tech Support & Online Security to secure retention.")
            elif 0.50 <= churn_probability < 0.65:
                st.warning("🟡 The Customer is at **Moderate-High Churn Risk:** Launch survey inquiries or offer feature upgrade trials to increase engagement.")
            elif 0.39 <= churn_probability < 0.50:
                st.info("🔵 The Customer is at **Elevated Churn Risk:** System threshold crossed. Engage with automated payment (Auto-Pay) setups to ease billing friction.")
            elif 0.25 <= churn_probability < 0.39:
                st.info("🟢 The Customer is at **Moderate Churn Risk:** Stable customer, but under-utilizing services. Propose loyalty bonuses or product updates.")
            else:
                st.success("✅ The Customer is at **Low Churn Risk:** Highly retained segment. Keep engaging with standard campaigns.")
        else:
            st.error(f"Error from server: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to FastAPI server at {API_URL}. Details: {e}")
