# models/model_trainer.py
import logging
import sys
import os
import pandas as pd
import numpy as np
import joblib
import MetaTrader5 as mt5 # <<< IMPORT ADDED HERE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from datetime import datetime

# --- Dynamic Import for Modules ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Assuming logging is setup by main entry point or we set it up here
    # If running standalone, need to ensure logging is configured first
    # from utils.logging_config import setup_logging # Uncomment if needed
    # setup_logging() # Uncomment if needed

    from config.config_loader import get_config
    # Import necessary functions from other modules
    from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5
    from trading_engine.engine import fetch_market_data # Reuse data fetching
    from features.feature_calculator import calculate_features # Reuse feature calc
except ImportError as e:
    print(f"FATAL ERROR: Could not import required modules in model_trainer.py. Error: {e}")
    # Add basic logging setup for error visibility if import fails early
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"Module import failed: {e}", exc_info=True)
    sys.exit(1)

# Get logger instance
logger = logging.getLogger(__name__)
# Ensure basic handler if none configured by other imports
if not logger.hasHandlers():
     handler = logging.StreamHandler(); handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
     logger.addHandler(handler); logger.setLevel(logging.INFO)
     logger.warning("Basic logging handler added as root logger seemed unconfigured in trainer.")


# --- Define Feature Columns for Training ---
# Must match the features we want the predictor to use later
# (Based on the list in predictor.py)
MODEL_FEATURE_COLS = [
    'returns', 'range', 'close_ratio', 'rsi_14', 'ema_20', 'ema_50', 'atr_14'
]


def create_target(df: pd.DataFrame, periods_ahead: int = 5) -> pd.DataFrame:
    """Creates a binary target: 1 if close price increases N periods ahead, else 0."""
    logger.debug(f"Creating target variable looking {periods_ahead} periods ahead.")
    df_target = df.copy()
    # Calculate future price change relative to current close
    future_return = df_target['close'].shift(-periods_ahead) / df_target['close'] - 1
    # Define target based on future return sign (can use threshold later)
    df_target['target'] = np.where(future_return > 0, 1, 0)
    # Drop rows where future price is unknown (last 'periods_ahead' rows)
    initial_rows = len(df_target)
    df_target.dropna(subset=['target'], inplace=True)
    rows_dropped = initial_rows - len(df_target)
    logger.debug(f"Dropped {rows_dropped} rows lacking future data for target creation.")
    return df_target


def train_and_save_model(
    symbol: str,
    timeframe, # MT5 timeframe constant
    num_bars_history: int = 5000, # Number of bars to fetch for training
    target_periods: int = 5,      # How many bars ahead to predict
    test_set_size: float = 0.2,   # Proportion of data for testing
    random_seed: int = 42
):
    """Fetches data, calculates features, trains model/scaler, evaluates, and saves."""
    logger.info(f"Starting model training process for {symbol}...")

    # --- 1. Fetch Data ---
    logger.info(f"Fetching {num_bars_history} bars of {symbol} ({timeframe})...")
    market_data_df = fetch_market_data(symbol, timeframe, num_bars_history)
    if market_data_df is None or market_data_df.empty:
        logger.error("Failed to fetch data for training. Aborting.")
        return False

    # --- 2. Calculate Features ---
    logger.info("Calculating features...")
    features_df = calculate_features(market_data_df)
    if features_df is None or features_df.empty:
        logger.error("Failed to calculate features for training. Aborting.")
        return False

    # --- 3. Create Target Variable ---
    logger.info(f"Creating target variable ({target_periods} periods ahead)...")
    data_with_target = create_target(features_df, periods_ahead=target_periods)
    if data_with_target.empty:
         logger.error("Dataset empty after target creation/dropna. Aborting.")
         return False

    # --- 4. Select Features and Target ---
    logger.info(f"Selecting features: {MODEL_FEATURE_COLS}")
    # Check if all expected feature columns exist AFTER potential NaNs from target creation
    if not all(col in data_with_target.columns for col in MODEL_FEATURE_COLS):
         missing = [col for col in MODEL_FEATURE_COLS if col not in data_with_target.columns]
         logger.error(f"Cannot train model. Missing required feature columns after processing: {missing}")
         return False

    X = data_with_target[MODEL_FEATURE_COLS]
    y = data_with_target['target']
    logger.info(f"Prepared X ({X.shape}) and y ({y.shape}) for training.")

    # --- 5. Split Data ---
    logger.info(f"Splitting data into train/test sets (Test size: {test_set_size:.0%})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_set_size, random_state=random_seed, stratify=y # Stratify helps with imbalanced classes
    )
    logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

    # --- 6. Scale Features ---
    logger.info("Fitting and saving scaler (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Fit ONLY on training data
    X_test_scaled = scaler.transform(X_test)       # Transform test data

    # Save the FITTED scaler
    scaler_filename = get_config('SCALER_FILENAME', 'scaler.pkl') # Use default if not in .env
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True) # Ensure models directory exists
    scaler_path = os.path.join(models_dir, scaler_filename)
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved successfully to: {scaler_path}")
    except Exception as e:
        logger.exception(f"Error saving scaler to {scaler_path}: {e}")
        return False # Abort if scaler can't be saved

    # --- 7. Train Model ---
    logger.info("Training model (GradientBoostingClassifier)...")
    # Basic parameters - can be tuned later
    model = GradientBoostingClassifier(
        n_estimators=100, # Fewer estimators for faster training initially
        learning_rate=0.1,
        max_depth=3,
        random_state=random_seed,
        verbose=0 # Set to 1 for training progress details
    )
    model.fit(X_train_scaled, y_train)

    # Save the TRAINED model
    model_filename = get_config('MODEL_FILENAME', 'model.pkl') # Use default if not in .env
    model_path = os.path.join(models_dir, model_filename)
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully to: {model_path}")
    except Exception as e:
        logger.exception(f"Error saving model to {model_path}: {e}")
        # Continue to evaluation even if saving fails? Or abort? Let's abort.
        return False

    # --- 8. Evaluate Model ---
    logger.info("Evaluating model performance on test set...")
    try:
        y_pred_test = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred_test)
        logger.info("--- Test Set Classification Report ---")
        # Log report line by line for better readability in log files
        for line in report.split('\n'):
             logger.info(line)
        # Also print to console
        print("\n--- Test Set Classification Report ---")
        print(report)
        logger.info("--------------------------------------")
    except Exception as e:
         logger.exception(f"Error during model evaluation: {e}")


    logger.info(f"Model training process completed for {symbol}.")
    return True


# --- Main execution block for running the trainer script ---
if __name__ == "__main__":
     # Setup logging if running this script directly
     try: from utils.logging_config import setup_logging; setup_logging()
     except ImportError: logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'); logger.info("Used basic logging config for trainer.")

     logger.info("="*10 + " Starting Model Training Script " + "="*10)

     # --- Configuration for Training ---
     train_symbol = 'EURUSD' # Which symbol to train on
     train_tf_str = 'TIMEFRAME_M5' # Timeframe as string
     # Use getattr safely with a default in case mt5 not imported or string invalid
     train_tf = getattr(mt5, train_tf_str, None)
     if train_tf is None:
          logger.error(f"Invalid MT5 timeframe string '{train_tf_str}' or MT5 module not available.")
          sys.exit(1) # Exit if timeframe is invalid

     history_bars = 5000 # How much data to use
     logger.info(f"Training parameters: Symbol={train_symbol}, Timeframe={train_tf_str}, History={history_bars} bars")


     # --- Initialize MT5 (Required for Data Fetching) ---
     mt5_ready = False
     logger.info("Initializing MT5 for training data...")
     if initialize_mt5():
         logger.info("MT5 initialized successfully for training.")
         mt5_ready = True
     else:
         logger.error("MT5 initialization failed. Cannot fetch training data.")

     # --- Run Training ---
     training_success = False
     if mt5_ready:
         try:
             training_success = train_and_save_model(
                 symbol=train_symbol,
                 timeframe=train_tf,
                 num_bars_history=history_bars
             )
         except Exception as train_e:
              logger.exception(f"An unexpected error occurred during training: {train_e}")
         finally:
              # --- Shutdown MT5 ---
              logger.info("Shutting down MT5 connection after training...")
              shutdown_mt5()
     else:
         logger.error("Skipping training because MT5 connection failed.")


     # --- Final Status ---
     if training_success:
          logger.info("Model training script finished successfully.")
          print("\nSUCCESS: Model and Scaler files should now be saved in the 'models' directory.")
     else:
          logger.error("Model training script finished with ERRORS.")
          print("\nFAILED: Model/Scaler files might not have been saved correctly. Check logs.")

     logger.info("="*10 + " Model Training Script Finished " + "="*10)