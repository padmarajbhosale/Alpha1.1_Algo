# database/create_tables.py
import sys
import os
import logging

# --- Setup Path and Imports ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.config_loader import get_config
    # Import the db_utils module itself
    from database import db_utils
    # Import the Base containing metadata from models
    from database.models import Base
    # Setup basic logging for this script
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)


def create_all_tables():
    """Creates all tables defined in models.py using the metadata from Base."""
    logger.info("Attempting to create database tables...")
    try:
        # Initialize DB connection - this should create/set db_utils.engine
        logger.info("Initializing database connection (if not already done)...")
        db_utils.initialize_database() # Call the function via the module

        # Check if the engine was successfully created within db_utils
        # Access the engine variable THROUGH the db_utils module name
        if db_utils.engine is None:
             logger.error("Failed to get database engine from db_utils module after initialization. Cannot create tables.")
             return False

        # Create all tables defined under models.Base metadata
        # Access the engine object via the imported module name for binding
        logger.info(f"Binding metadata to engine ({db_utils.engine.url.database} on {db_utils.engine.url.host}) and creating tables...")
        Base.metadata.create_all(bind=db_utils.engine)
        logger.info("Tables created successfully (or already exist).")
        return True

    except Exception as e:
        logger.exception(f"An error occurred during table creation: {e}")
        return False

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\nRunning script to create database tables defined in database/models.py...")
    db_name = get_config('DB_DATABASE', 'N/A')
    db_server = get_config('DB_SERVER', 'N/A')
    print(f"Target Database: '{db_name}' on Server: '{db_server}'")
    print("(Using connection details from .env file)")

    if create_all_tables():
        print("\nTable creation process completed successfully.")
        print(f"Please verify the 'trades_log' table exists in your '{db_name}' database using SSMS.")
    else:
        print("\nTable creation process FAILED. Check logs above for errors.")
    print("Script finished.")