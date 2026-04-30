import xgboost as xgb
from sklearn.ensemble import IsolationForest
import time

class FraudDetectionEngine:
    def __init__(self, scale_pos_weight):
        # XGBoost optimized for inference speed and high imbalance
        self.xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight, # Handles imbalance without slow SMOTE
            eval_metric='aucpr',               # Focus on Precision-Recall
            max_depth=4,                       # Kept shallow for millisecond latency
            learning_rate=0.1,
            n_estimators=100,
            n_jobs=-1
        )
        
        # Isolation Forest for Unsupervised Outlier Detection
        self.iso_forest = IsolationForest(
            n_estimators=100, 
            contamination=0.01, # We assume ~1% of ad-spend/transactions are fraud
            n_jobs=-1,
            random_state=42
        )

    def train_xgboost(self, X_train, y_train):
        self.xgb_model.fit(X_train, y_train)
        
    def train_isolation_forest(self, X_train):
        self.iso_forest.fit(X_train)

    def fast_predict(self, X):
        """Simulates millisecond-level ad-tech inference"""
        start_time = time.time()
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        iso_pred = self.iso_forest.predict(X)
        
        # Convert IsoForest (-1 is outlier, 1 is inlier) to 0-1 probability-like flag
        iso_flag = [1 if p == -1 else 0 for p in iso_pred]
        
        latency_ms = (time.time() - start_time) * 1000
        return xgb_prob, iso_flag, latency_ms