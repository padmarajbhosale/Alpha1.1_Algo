# main.py
import sys
import os
import time
import logging
import threading

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)

try: from utils.logging_config import setup_logging; setup_logging(); logger = logging.getLogger(__name__)
except Exception as log_e: print(f"FATAL ERROR logging setup: {log_e}"); logging.basicConfig(level=logging.ERROR); logging.critical(f"Logging setup exception: {log_e}", exc_info=True); sys.exit(1)

try:
    from shared_state import set_state
    from database.db_utils import initialize_database
    from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5, is_mt5_connected
    from trading_engine.engine import run_trading_loop
    # Import the NEW thread target function
    from telegram_interface.bot_handler import run_bot_in_thread # <<< IMPORT CHANGED
except ImportError as e: logger.critical(f"Module import failed: {e}", exc_info=True); sys.exit(1)


def main():
    logger.info("="*10 + " Starting Trading Bot Application " + "="*10)
    set_state("is_running", False); set_state("is_paused", False); set_state("mt5_connected", False); set_state("last_error", None)
    mt5_initialized_successfully = False

    try:
        logger.info("Initializing database..."); initialize_database(); logger.info("DB init successful.")
        logger.info("Initializing MT5..."); logger.info("Ensure MT5 running & Algo Trading enabled.")
        if initialize_mt5():
            logger.info("MT5 init successful."); set_state("mt5_connected", True); mt5_initialized_successfully = True
        else: logger.error("MT5 init failed. Exiting."); set_state("mt5_connected", False); return

        # --- Start Telegram Bot Thread (Using new target function) --- <<< TARGET CHANGED >>>
        logger.info("Starting Telegram bot thread...");
        telegram_thread = threading.Thread(target=run_bot_in_thread, name="TelegramBotThread", daemon=True)
        telegram_thread.start()
        logger.info("Telegram bot thread started.")
        # --- End Telegram Section ---

        logger.info("Starting trading engine loop..."); set_state("is_running", True)
        run_trading_loop() # Blocks here

    except KeyboardInterrupt: logger.info("Shutdown requested (Ctrl+C)."); set_state("is_running", False)
    except ConnectionError as e: logger.error(f"DB connection failed during init: {e}", exc_info=True); set_state("last_error", f"DB Init Error: {e}")
    except Exception as e: logger.exception(f"Critical error in main: {e}"); set_state("last_error", f"Critical Main Error: {e}"); set_state("is_running", False)
    finally:
        set_state("is_running", False)
        logger.info("Starting shutdown sequence...")
        if mt5_initialized_successfully:
            if is_mt5_connected(): shutdown_mt5()
            else: logger.warning("MT5 lost, skipping shutdown command.")
        else: logger.info("MT5 not initialized, skipping shutdown.")
        logger.info("="*10 + " Trading Bot Application Stopped " + "="*10)

if __name__ == "__main__":
    main()