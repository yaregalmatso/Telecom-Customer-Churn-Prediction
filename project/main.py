from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Telecom Customer Churn API",
    description="Production REST API serving the SMOTE-XGBoost classification model.",
    version="1.0.0"
)

# Dynamic robust path verification for Local Workspace, Docker, or Google Colab environments
POSSIBLE_MODEL_PATHS = [
    "./models/final_model_pipeline.joblib",
    "/app/models/final_model_pipeline.joblib",
    "/content/models/final_model_pipeline.joblib"
]

model = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f"Successfully loaded model pipeline from: {path}")
            break
        except Exception as e:
            print(f"Error loading from {path}: {e}")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {
        "status": "Online",
        "model_loaded": model is not None,
        "message": "Telecom Churn Prediction API is active and loaded!" if model else "API is online but model failed to load."
    }

@app.post("/predict")
def predict(customer: CustomerData):
    if not model:
         raise HTTPException(status_code=500, detail="Model weights file could not be verified/loaded on server.")

    # Convert incoming JSON schema to Pandas DataFrame
    input_df = pd.DataFrame([customer.dict()])

    # Feature Engineering transformations matching training workflow
    input_df["ValueRatio"] = input_df["TotalCharges"] / (input_df["tenure" ] + 1)
    input_df["TenureGroup"] = pd.cut(input_df["tenure"], bins=[0,12,24,36,72], labels=["New","Short","Medium","Long"])
    input_df["TenureGroup"] = input_df["TenureGroup"].astype(object).fillna("New")
    input_df["Monthly_to_Total_Ratio"] = input_df["MonthlyCharges"] / (input_df["TotalCharges"] + 1)
    input_df["Tenure_Charges_Interaction"] = input_df["tenure"] * input_df["MonthlyCharges"]
    input_df["Total_Charges_to_Monthly_Ratio"] = input_df["TotalCharges"] / (input_df["MonthlyCharges"] + 1)

    # Predict probabilities
    prob = float(model.predict_proba(input_df)[0, 1])
    prediction = 1 if prob >= 0.39 else 0

    return {
        "churn_probability": round(prob * 100, 2),
        "churn_prediction": prediction,
        "status": "HIGH CHURN RISK" if prediction == 1 else "LOW RISK (CUSTOMER RETAINED)"
    }
