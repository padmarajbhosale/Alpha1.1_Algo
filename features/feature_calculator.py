# features/feature_calculator.py
import pandas as pd
import numpy as np
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use exponential moving average (EMA) for average gain/loss - common RSI variant
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Prevent division by zero or very small numbers
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Ensure RSI is within 0-100 range (can sometimes slightly exceed due to precision)
    rsi = rsi.clip(0, 100)
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculates Moving Average Convergence Divergence (MACD)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    # Return as DataFrame for easy column naming and concat
    return pd.DataFrame({
        f'macd_{fast}_{slow}': macd,
        f'macd_signal_{signal}': macd_signal,
        f'macd_hist_{signal}': macd_hist
    })

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * float(std_dev))
    lower_band = sma - (std * float(std_dev))
    return pd.DataFrame({
        f'bb_middle_{period}': sma,
        f'bb_upper_{period}_{std_dev}': upper_band,
        f'bb_lower_{period}_{std_dev}': lower_band
    })

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    # Ensure inputs are numeric Series
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')
    close = pd.to_numeric(close, errors='coerce')

    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))

    # Combine the three components to find the True Range (TR)
    tr_df = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev})
    true_range = tr_df.max(axis=1, skipna=False) # Ensure NaN propagation if inputs have NaNs

    # Calculate ATR using Exponential Moving Average (common method)
    atr = true_range.ewm(com=period - 1, min_periods=period).mean()
    return atr


def calculate_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculates various technical indicators and adds them as columns to the DataFrame.

    Args:
        df: Input DataFrame with 'time', 'open', 'high', 'low', 'close'. 'tick_volume' is optional.

    Returns:
        DataFrame with added feature columns, or None if input is invalid or calculation fails.
        Drops rows with NaN values resulting from initial indicator calculations.
    """
    required_cols = ['time', 'open', 'high', 'low', 'close']
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Feature Calculator: Received invalid or empty DataFrame.")
        return None
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Feature Calculator: Input DataFrame missing required columns ({required_cols}). Found: {df.columns.tolist()}")
        return None

    logger.debug(f"Calculating features for DataFrame with shape {df.shape}")
    # Work on a copy to avoid modifying the original DataFrame passed to the function
    df_features = df.copy()

    try:
        # --- Ensure correct data types ---
        for col in ['open', 'high', 'low', 'close']:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

        # Drop rows where OHLC conversion failed (should not happen with MT5 data)
        df_features.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        if df_features.empty:
             logger.warning("DataFrame empty after ensuring numeric OHLC columns.")
             return None


        # --- Price derived features ---
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1)).fillna(0)
        df_features['range'] = df_features['high'] - df_features['low']
        # Close ratio within high-low range (handle zero range)
        df_features['close_ratio'] = ((df_features['close'] - df_features['low']) /
                                      df_features['range'].replace(0, 1e-10)).fillna(0.5).clip(0, 1)

        # --- Moving Averages ---
        df_features['sma_10'] = df_features['close'].rolling(window=10).mean()
        df_features['ema_20'] = df_features['close'].ewm(span=20, adjust=False).mean()
        df_features['ema_50'] = df_features['close'].ewm(span=50, adjust=False).mean()

        # --- Oscillators ---
        df_features['rsi_14'] = calculate_rsi(df_features['close'], period=14)
        macd_df = calculate_macd(df_features['close'])
        df_features = pd.concat([df_features, macd_df], axis=1)

        # --- Volatility ---
        df_features['atr_14'] = calculate_atr(df_features['high'], df_features['low'], df_features['close'], period=14)
        bb_df = calculate_bollinger_bands(df_features['close'])
        df_features = pd.concat([df_features, bb_df], axis=1)

        # --- Example Trend Feature (EMA Cross Diff) ---
        df_features['ema_diff'] = df_features['ema_20'] - df_features['ema_50']

        # --- Example Momentum Feature (Rate of Change) ---
        df_features['roc_10'] = df_features['close'].pct_change(periods=10) * 100 # As percentage

        # --- Time-based features (Example) ---
        # df_features['hour'] = df_features['time'].dt.hour
        # df_features['dayofweek'] = df_features['time'].dt.dayofweek # Monday=0

        # --- Drop rows with NaN values ---
        # This is crucial as indicators need lead-in periods.
        initial_rows = len(df_features)
        df_features.dropna(inplace=True)
        rows_dropped = initial_rows - len(df_features)
        if rows_dropped > 0:
             logger.debug(f"Dropped {rows_dropped} rows with NaN values after feature calculation.")

        if df_features.empty:
             logger.warning("DataFrame became empty after dropping NaNs post-feature calculation.")
             return None

        logger.info(f"Features calculated successfully. Output shape: {df_features.shape}")
        return df_features

    except Exception as e:
        logger.exception(f"An error occurred during feature calculation: {e}")
        return None


# --- Example Usage Section (for testing this file directly) ---
if __name__ == '__main__':
    # Basic logging setup for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    print("\n--- Testing Feature Calculation ---")
    # Create a more realistic sample DataFrame
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='5min'))
    close_prices = 1.1 + np.random.randn(100).cumsum() * 0.001
    data = {
        'time': dates,
        'open': close_prices - np.random.rand(100) * 0.0005,
        'high': close_prices + np.random.rand(100) * 0.001,
        'low': close_prices - np.random.rand(100) * 0.001,
        'close': close_prices,
        'tick_volume': np.random.randint(10, 100, size=100)
    }
    # Ensure OHLC consistency
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)

    sample_df = pd.DataFrame(data)
    print(f"Sample Input DataFrame shape: {sample_df.shape}")
    print("Sample Input Tail:\n", sample_df.tail(3))

    # Calculate features
    features_result_df = calculate_features(sample_df)

    if features_result_df is not None:
        print(f"\nOutput DataFrame with Features shape: {features_result_df.shape}")
        print("Output DataFrame Tail (showing calculated features):\n", features_result_df.tail(3))
        # Check for NaNs (should be False after dropna)
        print(f"\nAny NaN values remaining? {features_result_df.isnull().values.any()}")
        # print("\nColumns:", features_result_df.columns.tolist()) # Uncomment to see all columns
    else:
        print("\nFeature calculation failed.")

    print("\n--- Feature Calculation Test Complete ---") 
