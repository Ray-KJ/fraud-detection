import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import xgboost as xgb

app = FastAPI(title="Fraud Detection API")

@app.get("/")
def health_check():
    return {"status": "Model is loaded and ready for predictions"}

# 1. Load Artifacts
# We load these once when the app starts so it's fast!
model = xgb.XGBClassifier()
model.load_model("models/fraud_model.json")
scaler = joblib.load('models/scaler.pkl')

# 2. Define the input structure
class TransactionData(BaseModel):
    # Expecting: [Time, V1, V2, ... V28, Amount]
    features: List[float]

@app.post("/predict")
def predict(data: TransactionData):
    # Convert to DataFrame
    # Note: Column names must match the order used in training
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df = pd.DataFrame([data.features], columns=feature_names)
    
    # 3. Apply the SAME scaling we used in training
    # This is critical to avoid Training-Serving Skew
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    
    # 4. Make Prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    
    result = "FRAUD" if prediction[0] == 1 else "NOT FRAUD"
    
    return {
        "prediction": result,
        "fraud_probability": round(float(probability), 4),
        "status": "success"
    }