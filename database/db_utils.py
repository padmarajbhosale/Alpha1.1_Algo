# database/db_utils.py
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import sys
import os
import urllib.parse
from contextlib import contextmanager # <<< IMPORT ADDED HERE

# --- Dynamic Import for Configuration ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.config_loader import get_config
except ImportError as e:
    print(f"FATAL ERROR: Could not import get_config in db_utils.py. Error: {e}")
    sys.exit(1)

# --- Configure Logging ---
logger = logging.getLogger(__name__)
log_level = get_config('LOG_LEVEL', 'INFO').upper()
logger.setLevel(log_level)
if not logger.hasHandlers():
     handler = logging.StreamHandler(); handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
     logger.addHandler(handler)
     logger.warning("Basic logging handler added as root logger seemed unconfigured in db_utils.")


# --- Database Connection Setup ---
DATABASE_URL = None
engine = None # Global engine object, initialized by initialize_database()
SessionLocal = None # Global session factory, initialized by initialize_database()

def get_db_url() -> str:
    """Constructs the database URL for SQLAlchemy from configuration, handling Windows Auth."""
    driver = get_config('DB_DRIVER')
    server = get_config('DB_SERVER')
    database = get_config('DB_DATABASE')
    username = get_config('DB_USERNAME')
    password = get_config('DB_PASSWORD')
    trust_cert = get_config('TRUST_SERVER_CERTIFICATE', 'no') # Use TRUST_SERVER_CERTIFICATE from .env

    if not all([driver, server, database]):
        logger.error("DB config incomplete (Driver/Server/DB). Check .env.")
        raise ValueError("Incomplete DB configuration.")

    odbc_components = [ f"Driver={{{driver}}}", f"Server={server}", f"Database={database}",
                        "Encrypt=yes", f"TrustServerCertificate={trust_cert}", "Connection Timeout=30" ]

    if username: # SQL Server Auth
        logger.debug("Using SQL Server Authentication for DB URL.")
        if password is None: logger.warning("DB_PASSWORD not set for SQL Auth."); password = ""
        odbc_components.append(f"UID={username}"); odbc_components.append(f"PWD={password}")
    else: # Windows Auth
        logger.debug("Using Windows Authentication for DB URL.")
        odbc_components.append("Trusted_Connection=yes")

    try:
        odbc_connect_str = ";".join(odbc_components)
        params = urllib.parse.quote_plus(odbc_connect_str)
        db_url = f"mssql+pyodbc:///?odbc_connect={params}"
        logger.debug(f"Constructed DB URL using {'Windows Auth' if not username else 'SQL Auth'}.")
        return db_url
    except Exception as e: logger.exception(f"Error constructing database URL: {e}"); raise

def initialize_database():
    """Initializes the database engine and session maker using the URL from config."""
    global DATABASE_URL, engine, SessionLocal
    if engine is not None: logger.debug("Database already initialized."); return

    try:
        DATABASE_URL = get_db_url()
        engine = create_engine(DATABASE_URL, echo=False, pool_recycle=3600)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Database engine and session maker initialized successfully.")
    except ValueError as e: logger.error(f"DB init failed (Config Error?): {e}"); engine = None; SessionLocal = None; raise
    except SQLAlchemyError as e: logger.error(f"DB init failed (SQLAlchemy Error): {e}"); engine = None; SessionLocal = None; raise
    except Exception as e: logger.exception(f"Unexpected DB init error: {e}"); engine = None; SessionLocal = None; raise

# --- Functions to Use Database ---

@contextmanager # <<< DECORATOR ADDED HERE
def get_db_session():
    """Context manager for providing a database session. Ensures session is closed."""
    if SessionLocal is None:
        logger.error("DB session factory not initialized. Call initialize_database() first.")
        raise ConnectionError("SessionLocal is None. Ensure initialize_database() was called.")

    db = SessionLocal()
    logger.debug("DB Session created.") # Debug log
    try:
        yield db # Provide the session to the 'with' block
        # logger.debug("DB Session yield complete.") # Debug log
    except Exception as e:
        logger.error(f"Error during database session scope: {e}", exc_info=True)
        db.rollback() # Rollback on exception within 'with' block
        logger.info("DB Session rolled back due to error.")
        raise # Re-raise the exception after rollback
    finally:
        # Always close the session when exiting the 'with' block
        db.close()
        # logger.debug("DB Session closed.") # Debug log

def test_db_connection() -> bool:
    """Tests the database connection by executing 'SELECT 1'. Assumes engine is initialized."""
    # ... (test_db_connection remains the same) ...
    if engine is None: logger.error("DB engine not initialized."); return False
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")); scalar_result = result.scalar_one()
            logger.info(f"DB connection test successful. Result: {scalar_result}")
            return True
    except SQLAlchemyError as e: logger.error(f"DB connection test failed: {e}", exc_info=True); return False
    except Exception as e: logger.exception(f"Unexpected error during DB test: {e}"); return False


# --- Initialization and Testing Block ---
if __name__ == "__main__":
    # ... (if __name__ block remains the same) ...
    if not logging.getLogger().hasHandlers(): logging.basicConfig(level=logging.INFO)
    logger.info("--- Testing DB Utils (Standalone) ---")
    print("\nRunning script..."); db_name = get_config('DB_DATABASE', 'N/A'); db_server = get_config('DB_SERVER', 'N/A')
    print(f"Target DB: '{db_name}' on Server: '{db_server}'")
    try: initialize_database()
    except Exception as e: print(f"\nDB Initialization FAILED: {e}")
    else:
        if engine and SessionLocal:
            print("\nDB Initialization OK."); print("Attempting test query...");
            if test_db_connection(): print("--> DB connection test SUCCEEDED!")
            else: print("--> DB connection test FAILED.")
            print("\nAttempting to get session via context manager...")
            try:
                 with get_db_session() as test_db:
                      print("--> Successfully obtained DB session via context manager.")
                      print(f"    Session details: {test_db}")
            except Exception as e: print(f"--> FAILED to get session or error within 'with' block: {e}")
        else: print("\nDB Initialization FAILED (engine/SessionLocal is None).")
    print("\n--- DB Utils Test Complete ---")