# test_history.py
import MetaTrader5 as mt5
import time
import datetime
import os
import sys
import logging

# --- Setup Path and Imports ---
project_root = os.path.dirname(os.path.abspath(__file__)) # Assumes run from G:/Alpha1.1
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.config_loader import get_config
    # Need the connector functions
    from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5
    # Basic Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

if __name__ == "__main__":
    logger.info("--- Starting MT5 History Deals Test Script ---")
    logger.info("Ensure MT5 terminal is running and logged into the correct account.")
    logger.info("Using credentials from .env file for connection.")

    # Initialize MT5 using the function from connector
    if initialize_mt5():
        logger.info("MT5 Initialized successfully for history test.")
        try:
            # --- Define Time Range (e.g., last 24 hours using timestamps) ---
            now_ts = int(time.time())
            start_ts = now_ts - (24 * 60 * 60) # 24 hours ago in seconds epoch (UTC)
            logger.info(f"Attempting to fetch deals using mt5.history_deals_get from timestamp {start_ts} to {now_ts}...")

            # --- Call history_deals_get ---
            deals = mt5.history_deals_get(start_ts, now_ts)

            # --- Analyze Result ---
            if deals is None:
                logger.error(f"mt5.history_deals_get() returned None. Last Error: {mt5.last_error()}")
            elif len(deals) == 0:
                logger.warning("mt5.history_deals_get() returned 0 deals for the specified period (last 24h).")
            else:
                # THIS IS THE KEY OUTPUT WE NEED TO SEE
                logger.info(f"##### SUCCESS: mt5.history_deals_get() returned {len(deals)} deals! #####")

                # Print details of the first few deals for inspection
                num_to_print = min(len(deals), 10) # Print up to 10 deals
                logger.info(f"--- Details of first {num_to_print} deal(s) ---")
                for i in range(num_to_print):
                    deal = deals[i]
                    # Convert timestamp for readability
                    deal_time = datetime.datetime.fromtimestamp(deal.time, tz=datetime.timezone.utc)
                    # Log key details
                    logger.info(f"Deal #{i+1}: Ticket={deal.ticket}, Order={deal.order}, Time={deal_time}, Type={deal.type}, Entry={deal.entry}, Symbol={deal.symbol}, Price={deal.price}, Volume={deal.volume}, Profit={deal.profit}, Magic={deal.magic}")
                logger.info("--------------------------------------")
                if len(deals) > num_to_print:
                     logger.info(f"... and {len(deals) - num_to_print} more deals.")

        except Exception as e:
            logger.exception(f"An error occurred during history fetch test: {e}")
        finally:
            # --- Shutdown MT5 ---
            logger.info("Shutting down MT5 connection.")
            shutdown_mt5()
    else:
        logger.error("MT5 Initialization failed. Cannot perform history test.")

    logger.info("--- MT5 History Deals Test Script Finished ---") 
