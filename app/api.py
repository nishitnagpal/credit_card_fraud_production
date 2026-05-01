from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import time
import os
from src.features import engineer_features

app = FastAPI(title="Real-Time Anti-Fraud API")

# 1. Build the bulletproof absolute paths for BOTH models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_fraud_model.pkl")
ISO_MODEL_PATH = os.path.join(BASE_DIR, "models", "iso_forest_model.pkl")

# 2. Safely load BOTH models
if os.path.exists(XGB_MODEL_PATH) and os.path.exists(ISO_MODEL_PATH):
    xgb_model = joblib.load(XGB_MODEL_PATH)
    iso_forest = joblib.load(ISO_MODEL_PATH)
else:
    raise FileNotFoundError("CRITICAL: One or both models not found. Run training pipeline first.")

# 3. Define the expected incoming JSON payload
class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: int
    used_chip: int
    used_pin_number: int
    online_order: int

# 4. Define the outgoing JSON response (Now including the Anomaly Radar!)
class RiskResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    is_unsupervised_anomaly: bool
    action_taken: str
    financial_risk_mitigated: str
    latency_ms: float

# BUSINESS COST MATRIX CALIBRATION
CUSTOM_THRESHOLD = 0.15
COST_PER_FRAUD_LOSS = 100.00
COST_PER_FALSE_POSITIVE = 5.00

@app.post("/predict", response_model=RiskResponse)
def predict_fraud(payload: Transaction):
    # Start high-precision latency timer
    start_time = time.perf_counter()
    
    # Convert to DataFrame and apply feature engineering
    raw_data = pd.DataFrame([payload.dict()])
    processed_data = engineer_features(raw_data)
    
    # --- ENGINE 1: SUPERVISED XGBOOST ---
    fraud_prob = xgb_model.predict_proba(processed_data)[0][1]
    is_fraud = bool(fraud_prob >= CUSTOM_THRESHOLD)
    
    # --- ENGINE 2: UNSUPERVISED ISOLATION FOREST ---
    # Isolation Forest returns 1 for normal, -1 for anomaly
    anomaly_prediction = iso_forest.predict(processed_data)[0]
    is_anomaly = bool(anomaly_prediction == -1)
    
    # Formulate the business response
    if is_fraud:
        action = "Transaction Blocked"
        money_saved = f"${COST_PER_FRAUD_LOSS:.2f} (Expected Risk Mitigated)"
    elif is_anomaly and not is_fraud:
        action = "Flagged for Human Review (Concept Drift Warning)"
        money_saved = "$0.00 (Pending Review)"
    else:
        action = "Transaction Approved"
        money_saved = "$0.00"
        
    # Stop timer
    latency = round((time.perf_counter() - start_time) * 1000, 2)
    
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(float(fraud_prob), 4),
        "is_unsupervised_anomaly": is_anomaly,
        "action_taken": action,
        "financial_risk_mitigated": money_saved,
        "latency_ms": latency
    }