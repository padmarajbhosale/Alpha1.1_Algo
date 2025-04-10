import joblib
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

META_MODEL_PATH = os.getenv("META_MODEL_PATH", "models/model_META_xgboost.pkl")
META_SCALER_PATH = os.getenv("META_SCALER_PATH", "models/scaler_META.pkl")

_meta_model = None
_meta_scaler = None

def load_meta_model():
    global _meta_model, _meta_scaler
    if _meta_model is None or _meta_scaler is None:
        _meta_model = joblib.load(META_MODEL_PATH)
        _meta_scaler = joblib.load(META_SCALER_PATH)

def predict_meta_model(features_dict, threshold=0.6):
    load_meta_model()
    
    required_features = ['Regime', 'BOS_Signal', 'OB_Present', 'OB_Type_Encoded']
    features_array = np.array([[features_dict.get(f, 0) for f in required_features]])
    features_scaled = _meta_scaler.transform(features_array)

    proba = _meta_model.predict_proba(features_scaled)[0][1]  # class 1 = take trade
    should_trade = proba >= threshold
    return should_trade, float(proba)
