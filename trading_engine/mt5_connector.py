# trading_engine/mt5_connector.py
import MetaTrader5 as mt5
import logging
import time
import sys
import os

# --- Dynamic Import for Configuration & Logging ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

config_loaded = False
try:
    from config.config_loader import get_config
    config_loaded = True
    # We will get a logger instance, assuming logging is set up elsewhere (e.g., main.py)
    # If running this standalone, setup_logging needs to be called explicitly first.
except ImportError as e:
    print(f"FATAL ERROR: Could not import get_config. Error: {e}")
    # Define fallbacks if necessary for standalone testing without full structure
    # For now, exit if essential config loader isn't found
    sys.exit(1)

# Get a logger instance for this module
# Logging should be configured by the main application entry point.
# Basic config here for standalone testing if needed.
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Add basic handler if none configured yet
     handler = logging.StreamHandler()
     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
     handler.setFormatter(formatter)
     logger.addHandler(handler)
     logger.setLevel(logging.INFO) # Default level if not configured
     logger.warning("Basic logging handler added as root logger seemed unconfigured.")


def initialize_mt5() -> bool:
    """
    Initializes the MetaTrader 5 terminal connection and logs in.

    Returns:
        True if initialization and login are successful, False otherwise.
    """
    logger.info("Initializing MetaTrader 5 connection...")

    mt5_login_str = get_config('MT5_LOGIN')
    mt5_password = get_config('MT5_PASSWORD')
    mt5_server = get_config('MT5_SERVER')

    if not mt5_login_str or not mt5_password or not mt5_server:
        logger.error("MT5 credentials (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER) not fully configured in .env file.")
        return False

    # Convert login to integer
    try:
        login_int = int(mt5_login_str)
    except (ValueError, TypeError):
        logger.error(f"Invalid MT5_LOGIN '{mt5_login_str}'. It must be an integer.")
        return False

    # --- Initialize MT5 Terminal Connection ---
    logger.info("Attempting to initialize MT5 terminal...")
    # Ensure MT5 terminal app is running before calling initialize()
    try:
        # You can specify the path to terminal64.exe if it's not standard
        # terminal_path = get_config('MT5_TERMINAL_PATH') # Optional path from .env
        # initialized = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
        initialized = mt5.initialize()
    except Exception as init_e:
         logger.exception(f"Unexpected error during mt5.initialize(): {init_e}")
         return False


    if not initialized:
        logger.error(f"MT5 initialize() failed, error code: {mt5.last_error()}")
        logger.error("Ensure the MT5 terminal application (terminal64.exe) is running.")
        return False
    else:
        logger.info(f"MT5 terminal connection initialized successfully. Build: {mt5.terminal_info().build}")

    # --- Login to Trading Account ---
    logger.info(f"Attempting MT5 login for account {login_int} on server '{mt5_server}'...")
    try:
        authorized = mt5.login(login=login_int, password=mt5_password, server=mt5_server)
        if authorized:
            logger.info(f"MT5 login successful.")
            # Log account details
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Account Info: Login={account_info.login}, Name={account_info.name}, Balance={account_info.balance} {account_info.currency}, Server={account_info.server}")
                # Verify connected server matches expected server
                if account_info.server != mt5_server:
                     logger.warning(f"Connected to server '{account_info.server}' which differs from configured server '{mt5_server}'.")
            else:
                 logger.warning("Could not retrieve account info after login (mt5.account_info() returned None).")
            return True
        else:
            logger.error(f"MT5 login failed. Error code: {mt5.last_error()}")
            logger.error("Check login credentials, server name, and network connection in MT5 terminal.")
            # Shutdown connection if login fails
            mt5.shutdown()
            return False
    except Exception as e:
        logger.exception(f"An unexpected error occurred during MT5 login: {e}")
        mt5.shutdown() # Ensure shutdown on unexpected error
        return False


def shutdown_mt5():
    """Shuts down the MetaTrader 5 terminal connection."""
    logger.info("Shutting down MetaTrader 5 connection...")
    mt5.shutdown()
    logger.info("MT5 connection shut down.")

def is_mt5_connected() -> bool:
    """Checks if the MT5 connection is likely active (initialized and logged in)."""
    try:
        # Terminal info is basic check for initialization
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            # logger.debug("is_mt5_connected: terminal_info() returned None.")
            return False

        # Account info is better check for being logged in
        account_info = mt5.account_info()
        if account_info is None or account_info.login == 0:
            # logger.debug("is_mt5_connected: account_info() returned None or login is 0.")
            return False

        # If we have valid account info, assume connected
        return True
    except Exception as e:
        logger.error(f"Error checking MT5 connection status: {e}", exc_info=False)
        return False


# --- Example Usage Section (for testing this file directly) ---
if __name__ == "__main__":
    # Setup logging first IF running standalone
    # If this is imported by main.py, main.py should set up logging
    try:
        from utils.logging_config import setup_logging
        print("\n--- Setting up Logging for MT5 Connector Test ---")
        # Set level potentially higher for cleaner test output if desired
        # setup_logging(level='INFO')
        setup_logging()
    except ImportError:
        print("Could not import setup_logging. Using basic logging.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    logger.info("\n--- Initializing and Testing MT5 Connection ---")
    print("-" * 50)
    print("IMPORTANT REQUIREMENTS FOR THIS TEST:")
    print("1. Your MetaTrader 5 terminal application MUST be installed and RUNNING.")
    print("2. Your MT5 terminal must be logged into an account.")
    print("3. 'Allow Algo Trading' MUST be enabled in MT5 (Tools -> Options -> Expert Advisors).")
    print("4. Correct MT5_LOGIN, MT5_PASSWORD, MT5_SERVER MUST be set in G:/Alpha1.1/.env")
    print("-" * 50)


    # Attempt to initialize and login
    connection_success = initialize_mt5()

    if connection_success:
        logger.info("MT5 Initialization and Login SUCCEEDED in test.")

        # Perform some checks if connected
        print("\n--- Checking Connection Status ---")
        if is_mt5_connected():
            logger.info("is_mt5_connected() returned True.")
            term_info = mt5.terminal_info()
            acc_info = mt5.account_info()
            if term_info:
                print(f"Terminal Info: Build {term_info.build}, Path='{term_info.path}'")
            if acc_info:
                print(f"Account Info: Login={acc_info.login}, Name='{acc_info.name}', Server='{acc_info.server}', Balance={acc_info.balance} {acc_info.currency}")
        else:
             logger.warning("is_mt5_connected() returned False unexpectedly after successful initialization/login.")

        # Remember to shut down
        print("\n--- Shutting Down MT5 Connection ---")
        shutdown_mt5()
    else:
        logger.error("MT5 Initialization or Login FAILED during test. Please check:")
        logger.error("1. MT5 terminal is running and logged in.")
        logger.error("2. 'Allow Algo Trading' is enabled in MT5 options.")
        logger.error("3. Credentials/Server in .env file match the running terminal.")
        logger.error("4. Any specific error messages logged above.")

    print("\n--- MT5 Connector Test Complete ---") 
