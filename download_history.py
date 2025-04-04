# download_history.py
import MetaTrader5 as mt5
import pandas as pd
import datetime
import argparse
import os
import sys
import logging

# --- Setup Path for Imports ---
# Adds the project root directory to the path so we can import other modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Project Modules ---
try:
    # We only need the connector functions for this script
    from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5
except ImportError as e:
    print(f"FATAL ERROR: Could not import mt5_connector. Ensure it exists and __init__.py files are present. Error: {e}")
    sys.exit(1)

# --- Basic Logging Setup for this Script ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- MT5 Timeframe Mapping ---
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}

def download_data(symbol: str, timeframe_str: str, start_date: str, end_date: str, output_dir: str):
    """Downloads historical data from MT5 and saves it to a CSV file."""

    logger.info(f"Starting data download for {symbol} ({timeframe_str}) from {start_date} to {end_date}")

    # --- Validate Inputs ---
    tf_upper = timeframe_str.upper()
    if tf_upper not in TIMEFRAME_MAP:
        logger.error(f"Invalid timeframe string: {timeframe_str}. Valid options: {list(TIMEFRAME_MAP.keys())}")
        return False
    timeframe = TIMEFRAME_MAP[tf_upper]

    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        # Add time component to end_date to ensure full day is included
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD.")
        return False

    # --- Ensure Output Directory Exists ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    except OSError as e:
        logger.error(f"Could not create output directory '{output_dir}': {e}")
        return False

    # --- Construct Filename ---
    # Use consistent naming convention
    filename = f"{symbol}_{tf_upper}_{start_date}_to_{end_date}.csv"
    filepath = os.path.join(output_dir, filename)
    logger.info(f"Data will be saved to: {filepath}")

    # --- Initialize MT5 ---
    # Requires MT5 terminal running and logged in!
    logger.info("Initializing MT5 connection...")
    if not initialize_mt5():
        logger.error("Failed to initialize MT5 connection. Ensure terminal is running and configured.")
        return False
    logger.info("MT5 connection successful.")

    # --- Fetch Data ---
    rates = None
    try:
        logger.info(f"Requesting rates using copy_rates_range...")
        # Use copy_rates_range for specific date periods
        rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)

        if rates is None:
            logger.error(f"Failed to fetch rates for {symbol}. mt5.copy_rates_range returned None. MT5 Error: {mt5.last_error()}")
            return False # Return False on failure

        if len(rates) == 0:
            logger.warning(f"No rates data returned for {symbol} in the specified date range.")
            # Still return True, as the operation succeeded but yielded no data
            return True

        logger.info(f"Successfully retrieved {len(rates)} bars for {symbol}.")

        # --- Process and Save Data ---
        rates_df = pd.DataFrame(rates)
        # Convert time to datetime objects (MT5 time is usually UTC based on server)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
        # Select standard columns (case might vary, adjust if needed based on MT5 version/broker)
        # Standard MT5 columns: time, open, high, low, close, tick_volume, spread, real_volume
        standard_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume'] # Adjust if real_volume is preferred/available
        rates_df = rates_df[standard_cols]

        rates_df.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logger.info(f"Data successfully saved to {filepath}")
        return True # Indicate success

    except Exception as e:
        logger.exception(f"An error occurred during data fetch or processing: {e}")
        return False # Indicate failure

    finally:
        # --- Shutdown MT5 ---
        logger.info("Shutting down MT5 connection.")
        shutdown_mt5()


# --- Main execution block when script is run directly ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical market data from MetaTrader 5.")
    parser.add_argument("-s", "--symbol", required=True, help="Trading symbol (e.g., EURUSD, XAUUSD)")
    parser.add_argument("-tf", "--timeframe", required=True, help=f"Timeframe (e.g., M1, M5, H1, D1). Options: {list(TIMEFRAME_MAP.keys())}")
    parser.add_argument("-start", "--startdate", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("-end", "--enddate", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("-o", "--outputdir", default="./data", help="Output directory for CSV file (default: ./data)")

    args = parser.parse_args()

    # Run the download function with parsed arguments
    success = download_data(
        symbol=args.symbol.upper(), # Ensure symbol is uppercase
        timeframe_str=args.timeframe,
        start_date=args.startdate,
        end_date=args.enddate,
        output_dir=args.outputdir
    )

    if success:
        logger.info("Download process finished successfully.")
        sys.exit(0) # Exit with success code
    else:
        logger.error("Download process finished with errors.")
        sys.exit(1) # Exit with error code