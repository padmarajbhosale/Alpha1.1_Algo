 
# models/predictor.py
import joblib # For loading .pkl files
import os
import sys
import logging
import pandas as pd
import numpy as np

# --- Dynamic Import for Configuration ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.config_loader import get_config
except ImportError as e:
    print(f"FATAL ERROR: Could not import get_config in predictor.py. Error: {e}")
    sys.exit(1)

# Get logger for this module
logger = logging.getLogger(__name__)
# Add basic handler if needed (e.g., if run standalone)
if not logger.hasHandlers():
    handler = logging.StreamHandler(); handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler); logger.setLevel(logging.INFO)
    logger.warning("Basic logging handler added as root logger seemed unconfigured in predictor.")


# --- Define Feature Columns Expected by the Trained Model ---
# Must match the features used during training in model_trainer.py
MODEL_FEATURE_COLS = [
    'returns', 'range', 'close_ratio', 'rsi_14', 'ema_20', 'ema_50', 'atr_14'
]


def load_model_and_scaler() -> tuple[object | None, object | None]:
    """Loads the scikit-learn model and scaler from files specified in config."""
    model_filename = get_config('MODEL_FILENAME')
    scaler_filename = get_config('SCALER_FILENAME')
    models_dir = os.path.join(project_root, 'models') # Standard models directory

    if not model_filename or not scaler_filename:
        logger.error("MODEL_FILENAME or SCALER_FILENAME not set in .env file.")
        return None, None

    model_path = os.path.join(models_dir, model_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)

    model = None
    scaler = None

    # Load Scaler
    try:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded successfully from: {scaler_path}")
        else:
            logger.error(f"Scaler file not found at: {scaler_path}")
    except Exception as e:
        logger.exception(f"Error loading scaler from {scaler_path}: {e}")

    # Load Model
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
        else:
            logger.error(f"Model file not found at: {model_path}")
    except Exception as e:
        logger.exception(f"Error loading model from {model_path}: {e}")

    # Return loaded objects (could be None if loading failed)
    return model, scaler


def make_prediction(model, scaler, features_df: pd.DataFrame) -> tuple[int | None, float | None]:
    """
    Makes a prediction using the loaded model and scaler on the latest features.

    Args:
        model: The loaded scikit-learn model object.
        scaler: The loaded scikit-learn scaler object.
        features_df: DataFrame containing calculated features (including history).

    Returns:
        A tuple containing:
          - prediction (int: 0 or 1) or None if prediction fails.
          - confidence (float: probability of the predicted class) or None if fails.
    """
    if model is None or scaler is None:
        logger.error("Model or Scaler not loaded. Cannot make prediction.")
        return None, None
    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        logger.error("Received invalid or empty DataFrame for prediction.")
        return None, None

    try:
        # 1. Get the latest row of data (ensure it has necessary features)
        latest_data = features_df.iloc[-1:] # Keep as DataFrame row

        # 2. Select ONLY the feature columns the model expects
        if not all(col in latest_data.columns for col in MODEL_FEATURE_COLS):
             missing = [col for col in MODEL_FEATURE_COLS if col not in latest_data.columns]
             logger.error(f"Cannot make prediction. Missing required feature columns: {missing}")
             return None, None

        latest_features = latest_data[MODEL_FEATURE_COLS]
        logger.debug(f"Features selected for prediction (shape {latest_features.shape}): {MODEL_FEATURE_COLS}")

        # Check for NaNs in the selected features before scaling
        if latest_features.isnull().values.any():
             logger.warning(f"NaN values found in latest features before scaling. Prediction may be unreliable. Features:\n{latest_features}")
             # Cannot proceed if NaNs exist, scaler/model will likely fail
             return None, None

        # 3. Scale the features
        scaled_features = scaler.transform(latest_features)
        logger.debug(f"Features scaled successfully. Shape: {scaled_features.shape}")

        # 4. Make prediction
        prediction = model.predict(scaled_features)[0]
        prediction = int(prediction) # Ensure integer type

        # 5. Get probability (confidence)
        probabilities = model.predict_proba(scaled_features)[0]
        # Confidence is the probability of the PREDICTED class
        confidence = probabilities[prediction]
        confidence = float(confidence) # Ensure float type

        # Log the outcome before returning
        # logger.info(f"Prediction: Class={prediction}, Confidence={confidence:.4f}") # Moved logging to engine
        return prediction, confidence

    except Exception as e:
        logger.exception(f"An error occurred during make_prediction: {e}")
        return None, None


# --- Example Usage Section (for testing this file directly) ---
if __name__ == '__main__':
     # Setup basic logging for testing
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

     print("\n--- Testing Model/Scaler Loading and Prediction ---")
     print("Ensure model/scaler .pkl files are in G:/Alpha1.1/models/")
     print("Ensure MODEL_FILENAME/SCALER_FILENAME are set in G:/Alpha1.1/.env")

     # Load model and scaler
     loaded_model, loaded_scaler = load_model_and_scaler()

     if loaded_model and loaded_scaler:
         print("\nModel and Scaler loaded successfully.")

         # Create sample features for prediction test
         print("\nCreating sample features for prediction test...")
         # Ensure sample data matches MODEL_FEATURE_COLS
         sample_feature_data = {col: [np.random.rand()] for col in MODEL_FEATURE_COLS} # Single row
         sample_features_df = pd.DataFrame(sample_feature_data)
         # Add a dummy 'time' column to simulate structure (not used in prediction)
         sample_features_df['time'] = pd.Timestamp.now()
         print("Sample features DataFrame (1 row) columns:", sample_features_df.columns.tolist())
         print(sample_features_df[MODEL_FEATURE_COLS])

         # Make prediction on sample data
         print("\nAttempting prediction on sample features...")
         pred, conf = make_prediction(loaded_model, loaded_scaler, sample_features_df)

         if pred is not None and conf is not None:
             print(f"\n--> Test Prediction SUCCEEDED: Class={pred}, Confidence={conf:.4f}")
         else:
             print("\n--> Test Prediction FAILED. Check logs for errors.")
     else:
         print("\nModel or Scaler loading FAILED. Cannot perform prediction test.")

     print("\n--- Predictor Test Complete ---")