import pandas as pd
import numpy as np
import os
import sys

# --- Library Check ---
try:
    import talib
    print("Using TA-Lib for indicator calculation.")
    USE_TALIB = True
except ImportError:
    try:
        import pandas_ta as pta
        print("TA-Lib not found. Using pandas_ta for indicator calculation.")
        USE_TALIB = False
    except ImportError:
        print("ERROR: Neither TA-Lib nor pandas_ta found. Please install one (e.g., 'pip install TA-Lib pandas_ta')")
        sys.exit()

# --- Configuration ---
DATA_FILE = r"G:\Alpha1.1\data\EURUSD_M5_2024-04-01_to_2025-04-02.csv"
OUTPUT_FILE = r"G:\Alpha1.1\data\EURUSD_M5_2024_2025_RegimeLabeled_H1.csv"
REGIME_TIMEFRAME = '1h'  # lowercase 'h' avoids future deprecation warning
EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200
ADX_PERIOD = 14
ADX_THRESHOLD = 20

# --- 1. Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df_m5 = pd.read_csv(DATA_FILE, parse_dates=['time'], dayfirst=True)
    df_m5.rename(columns={
        'time': 'Timestamp',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)

    if 'Timestamp' in df_m5.columns:
        df_m5.set_index('Timestamp', inplace=True)
    else:
        raise KeyError("'Timestamp' column not found after renaming 'time'. Check original CSV.")

    print("Data loaded successfully.")
    print(f"Columns: {df_m5.columns.tolist()}")
    print(f"Data shape: {df_m5.shape}")
    print(f"Time range: {df_m5.index.min()} to {df_m5.index.max()}")

except FileNotFoundError:
    print(f"ERROR: File not found at {DATA_FILE}. Please check the path.")
    sys.exit()
except KeyError as e:
    print(f"ERROR: Column issue during loading/renaming. Error: {e}")
    sys.exit()
except Exception as e:
    print(f"ERROR: Could not load or process data. Check CSV format/content. Error: {e}")
    sys.exit()

# --- 2. Data Validation ---
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in df_m5.columns for col in required_cols):
    print(f"ERROR: Missing required columns. Found: {df_m5.columns.tolist()}. Required: {required_cols}")
    sys.exit()
if df_m5.empty:
    print(f"ERROR: DataFrame is empty. Check file content: {DATA_FILE}")
    sys.exit()

# --- 3. Ensure DatetimeIndex ---
if not isinstance(df_m5.index, pd.DatetimeIndex):
    print("Converting index to DatetimeIndex for resampling...")
    df_m5.index = pd.to_datetime(df_m5.index, errors='coerce')

    if df_m5.index.isnull().any():
        print("❌ ERROR: Some timestamps could not be converted. Check your CSV format.")
        print(df_m5[df_m5.index.isnull()].head())
        sys.exit()

# --- 4. Resample to H1 for Regime Detection ---
print(f"Resampling M5 data to {REGIME_TIMEFRAME}...")
ohlc_dict = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}
df_m5 = df_m5.sort_index()
df_h1 = df_m5.resample(REGIME_TIMEFRAME, label='right', closed='right').agg(ohlc_dict).dropna()

if df_h1.empty:
    print(f"ERROR: Resampled DataFrame is empty. Check time range or input granularity.")
    sys.exit()
print(f"Resampling complete. {len(df_h1)} {REGIME_TIMEFRAME} bars created.")

# --- 5. Indicator Calculation ---
print(f"Calculating indicators on {REGIME_TIMEFRAME} data...")
min_bars_needed = max(EMA_SLOW_PERIOD, ADX_PERIOD) + 5
if len(df_h1) < min_bars_needed:
    print(f"WARNING: Only {len(df_h1)} bars, less than the recommended {min_bars_needed} for stable indicators.")

try:
    if USE_TALIB:
        df_h1['EMA_Fast'] = talib.EMA(df_h1['Close'], timeperiod=EMA_FAST_PERIOD)
        df_h1['EMA_Slow'] = talib.EMA(df_h1['Close'], timeperiod=EMA_SLOW_PERIOD)
        df_h1['ADX'] = talib.ADX(df_h1['High'], df_h1['Low'], df_h1['Close'], timeperiod=ADX_PERIOD)
    else:
        df_h1.ta.ema(length=EMA_FAST_PERIOD, append=True, col_names=('EMA_Fast'))
        df_h1.ta.ema(length=EMA_SLOW_PERIOD, append=True, col_names=('EMA_Slow'))
        adx_df = df_h1.ta.adx(length=ADX_PERIOD)
        df_h1['ADX'] = adx_df[f'ADX_{ADX_PERIOD}']

    before_drop = len(df_h1)
    df_h1.dropna(subset=['EMA_Fast', 'EMA_Slow', 'ADX'], inplace=True)
    print(f"Dropped {before_drop - len(df_h1)} rows with NaN indicators.")

except Exception as e:
    print(f"ERROR: Indicator calculation failed. Error: {e}")
    sys.exit()

# --- 6. Apply Regime Logic ---
print("Applying regime rules...")
conditions = [
    (df_h1['Close'] > df_h1['EMA_Slow']) & (df_h1['EMA_Fast'] > df_h1['EMA_Slow']) & (df_h1['ADX'] > ADX_THRESHOLD),
    (df_h1['Close'] < df_h1['EMA_Slow']) & (df_h1['EMA_Fast'] < df_h1['EMA_Slow']) & (df_h1['ADX'] > ADX_THRESHOLD),
]
choices = ['Bullish', 'Bearish']
df_h1['Regime'] = np.select(conditions, choices, default='Ranging')
print("Regime labeling complete.")

# --- 7. Map Regimes to M5 ---
print(f"Mapping {REGIME_TIMEFRAME} regimes back to M5 data...")
df_h1 = df_h1.sort_index()
df_m5_labeled = pd.merge_asof(df_m5.sort_index(), df_h1[['Regime']], left_index=True, right_index=True, direction='backward')
df_m5_labeled['Regime'].fillna(method='bfill', inplace=True)
print("Mapping complete.")

# --- 8. Save Results ---
print(f"Preparing to save labeled M5 data to {OUTPUT_FILE}...")

output_dir = os.path.dirname(OUTPUT_FILE)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

try:
    df_m5_labeled.to_csv(OUTPUT_FILE, index=True)
    print(f"✅ Labeled M5 data saved to: {OUTPUT_FILE}")
    print(df_m5_labeled[['Open', 'High', 'Low', 'Close', 'Regime']].tail(5))
except Exception as e:
    print(f"❌ ERROR saving M5 data. Error: {e}")
    sys.exit()

# --- 9. Optional: Save H1 Regime Data ---
h1_output_file = OUTPUT_FILE.replace(".csv", "_H1_Regimes.csv")
try:
    df_h1[['Open', 'High', 'Low', 'Close', 'EMA_Fast', 'EMA_Slow', 'ADX', 'Regime']].to_csv(h1_output_file)
    print(f"📁 H1 regime data also saved to: {h1_output_file}")
except Exception as e:
    print(f"⚠️ WARNING: Could not save H1 regime data. Error: {e}")

# --- Done ---
print("🎯 Regime labeling process completed successfully.")
