import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --- Configuration ---
DATA_FILE = r"G:\Alpha1.1\data\EURUSD_M5_2024_2025_RegimeLabeled_H1_H1_Regimes.csv"
PLOT_DIR = r"G:\Alpha1.1\plots"
START_DATE = "2024-07-01"
END_DATE = "2024-09-30"
PLOT_FILENAME = "regime_plot_2024_Q3.png"

# --- 1. Load Data ---
print(f"Loading H1 regime data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"], index_col="Timestamp")
    print("Data loaded successfully.")
    print(f"Columns available: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ ERROR: Failed to load file. {e}")
    exit()

# --- 2. Filter by Date Range ---
print(f"Filtering data for period: {START_DATE} to {END_DATE}")
df = df.loc[START_DATE:END_DATE]

if df.empty:
    print("❌ ERROR: No data available in the selected date range.")
    exit()

# --- 3. Plotting ---
print("Preparing plot...")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5)

# Define color mapping
regime_colors = {
    'Bullish': 'green',
    'Bearish': 'red',
    'Ranging': 'gray'
}

print("Adding regime background shading...")
current_regime = None
start_idx = None

for i in range(len(df)):
    regime = df['Regime'].iloc[i]
    timestamp = df.index[i]

    if current_regime is None:
        current_regime = regime
        start_idx = timestamp

    elif regime != current_regime or i == len(df) - 1:
        end_idx = timestamp
        ax.axvspan(start_idx, end_idx, color=regime_colors.get(current_regime, 'gray'), alpha=0.15)
        current_regime = regime
        start_idx = timestamp

# --- 4. Formatting ---
ax.set_title(f"EUR/USD H1 Close Price with Regime Zones ({START_DATE} to {END_DATE})", fontsize=14)
ax.set_ylabel("Price")
ax.set_xlabel("Date")
ax.legend(["Close Price"])
ax.grid(True)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# --- 5. Save Plot ---
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created plots directory: {PLOT_DIR}")

plot_path = os.path.join(PLOT_DIR, PLOT_FILENAME)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"✅ Plot saved to: {plot_path}")
