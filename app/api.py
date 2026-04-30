from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import time
import joblib
import os
from src.features import engineer_features

app = FastAPI(title="Adjoe Anti-Fraud Microservice")

# Load pre-trained model on startup
MODEL_PATH = "models/xgb_fraud_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: int
    used_chip: int
    used_pin_number: int
    online_order: int

@app.post("/predict")
def predict_fraud(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    start_time = time.time()
    
    # Convert to DataFrame
    raw_data = pd.DataFrame([tx.dict()])
    
    # MUST apply the same feature engineering used in training
    processed_data = engineer_features(raw_data)
    
    # Predict using the real model
    prob = model.predict_proba(processed_data)[:, 1][0]
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(prob > 0.80), # Strict threshold to avoid false positives
        "inference_latency_ms": round(latency_ms, 2),
        "status": "success"
    }