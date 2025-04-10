# features/feature_calculator.py
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta # <<< Import pandas-ta

logger = logging.getLogger(__name__)

def calculate_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculates technical features for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 'time', 'open', 'high', 'low', 'close'.
                           'time' might be index or column.

    Returns:
        pd.DataFrame | None: DataFrame with added features, or None if error.
    """
    logger.debug(f"Calculating features for DataFrame with shape {df.shape}")
    required_cols = ['open', 'high', 'low', 'close']
    # Check if 'time' is a column, if not, assume it's the index
    if 'time' not in df.columns and df.index.name == 'time':
         df_calc = df.reset_index() # Work with time as a column for safety
         logger.debug("Time was index, reset for calculation.")
    elif 'time' in df.columns:
         df_calc = df.copy() # Work on a copy
    else:
         logger.error("Input DataFrame missing 'time' column or index.")
         return None

    if not all(col in df_calc.columns for col in required_cols):
        logger.warning(f"Feature Calculator: Input DataFrame missing required columns ({required_cols}). Found: {df_calc.columns.tolist()}")
        return None

    try:
        # Ensure 'time' is datetime type if it's a column
        if 'time' in df_calc.columns:
             df_calc['time'] = pd.to_datetime(df_calc['time'])

        # 1. Basic Price Features
        df_calc['returns'] = df_calc['close'].pct_change()
        df_calc['range'] = df_calc['high'] - df_calc['low']
        df_calc['close_ratio'] = (df_calc['close'] - df_calc['low']) / (df_calc['high'] - df_calc['low']) # Normalize close within range
        df_calc['close_ratio'] = df_calc['close_ratio'].fillna(0.5) # Handle bars with zero range

        # 2. Standard Indicators (Using pandas_ta)
        # RSI
        df_calc.ta.rsi(length=14, append=True) # Appends 'RSI_14'

        # EMAs
        df_calc.ta.ema(length=20, append=True) # Appends 'EMA_20'
        df_calc.ta.ema(length=50, append=True) # Appends 'EMA_50'

        # ATR
        df_calc.ta.atr(length=14, append=True) # Appends 'ATRr_14' (note the 'r')

        # --- NEW INDICATORS --- <<< ADDED HERE >>>
        # Bollinger Bands (length 20, std dev 2)
        # Appends: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        logger.debug("Calculating Bollinger Bands...")
        df_calc.ta.bbands(close='close', length=20, std=2, append=True)

        # MACD (standard 12, 26, 9 periods)
        # Appends: MACD_12_26_9, MACDh_12_26_9 (Histogram), MACDs_12_26_9 (Signal)
        logger.debug("Calculating MACD...")
        df_calc.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        # --- End New Indicators ---

        # 3. Drop rows with NaN values created by indicator lookback periods
        initial_len = len(df_calc)
        df_calc.dropna(inplace=True)
        final_len = len(df_calc)
        logger.debug(f"Dropped {initial_len - final_len} rows with NaN values after feature calculation.")

        # Set time back as index if it was originally
        if df.index.name == 'time':
             df_calc = df_calc.set_index('time')
             logger.debug("Set time back to index.")

        logger.info(f"Features calculated successfully. Output shape: {df_calc.shape}")
        return df_calc

    except Exception as e:
        logger.exception(f"An error occurred during feature calculation: {e}")
        return None

# --- Test Block ---
if __name__ == '__main__':
    # Example usage (requires a sample data CSV)
    import os # Moved import here for standalone test
    logging.basicConfig(level=logging.INFO) # Basic config for standalone test
    logger.info("--- Testing Feature Calculator Standalone ---")
    # Adjusted path assuming execution from G:\Alpha1.1\features
    sample_data_path = 'G:\\Alpha1.1\\data\\EURUSD\\EURUSD_M5_2024-04-01_to_2025-04-02.csv' # Use one of your actual data files
    if os.path.exists(sample_data_path):
         df_raw = pd.read_csv(sample_data_path, parse_dates=['time'], index_col='time')
         logger.info(f"Loaded sample data shape: {df_raw.shape}")
         df_features = calculate_features(df_raw.head(200)) # Calculate on first 200 rows
         if df_features is not None:
             logger.info("Features calculated for sample data.")
             print(df_features.tail())
             print("\nColumns:")
             print(df_features.columns.tolist())
             logger.info(f"Final shape: {df_features.shape}")
         else:
             logger.error("Feature calculation failed for sample data.")
    else:
         logger.warning(f"Sample data file not found at {sample_data_path}. Cannot run standalone test.")