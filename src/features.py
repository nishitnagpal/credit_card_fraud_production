import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Transforms raw transaction data into production-ready features.
    """
    df = df.copy()
    
    # 1. 'Ad-Tech' Geospatial Anomaly Feature
    # Simulates checking if a mobile SDK location differs drastically from normal
    # (Using the 'distance_from_home' and 'distance_from_last_transaction' columns)
    df['is_location_anomaly'] = np.where(
        (df['distance_from_home'] > df['distance_from_home'].quantile(0.95)) | 
        (df['distance_from_last_transaction'] > df['distance_from_last_transaction'].quantile(0.95)), 
        1, 0
    )
    
    # 2. Velocity / Behavior Ratio Feature
    # Ratio of current purchase to their historical median
    # High ratios often indicate account takeover
    df['purchase_velocity_risk'] = df['ratio_to_median_purchase_price'].apply(
        lambda x: 1 if x > 5.0 else (0.5 if x > 2.0 else 0)
    )
    
    return df