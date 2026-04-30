from src.data_loader import DataLoader
from src.features import engineer_features
from src.models import FraudDetectionEngine
from src.evaluate import evaluate_business_cost
from sklearn.model_selection import train_test_split
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("--- Starting Anti-Fraud Pipeline ---")
    
    # 1. Ingest
    loader = DataLoader("dhanushnarayananr/credit-card-fraud")
    df = loader.fetch_training_data()
    loader.validate_schema(df)
    
    # 2. Feature Engineering
    logging.info("Engineering velocity and spatial features...")
    df_processed = engineer_features(df)
    
    # 3. Prepare for Training
    # UPDATED: Dropping 'fraud' instead of 'Class'
    X = df_processed.drop(columns=['fraud'])
    y = df_processed['fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 4. Train Models
    # Calculate scale_pos_weight dynamically for XGBoost
    imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    engine = FraudDetectionEngine(scale_pos_weight=imbalance_ratio)
    
    logging.info("Training XGBoost Classifier...")
    engine.train_xgboost(X_train, y_train)
    
    logging.info("Training Unsupervised Isolation Forest...")
    engine.train_isolation_forest(X_train)
    
    # 5. Evaluate
    logging.info("Evaluating Business Impact on Holdout Set...")
    xgb_probs, _, _ = engine.fast_predict(X_test)
    xgb_preds = (xgb_probs > 0.5).astype(int) 
    
    evaluate_business_cost(y_test, xgb_preds, xgb_probs)
    
    # 6. Save the Models
    logging.info("Saving models for production API...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(engine.xgb_model, "models/xgb_fraud_model.pkl")
    joblib.dump(engine.iso_forest, "models/iso_forest_model.pkl")
    
    logging.info("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()