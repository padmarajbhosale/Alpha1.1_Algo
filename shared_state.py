# shared_state.py
import threading
import logging

logger = logging.getLogger(__name__)

# Shared state dictionary - stores current bot status
bot_state = {
    "is_running": False,
    "is_paused": False,
    "mt5_connected": False,
    "last_error": None,
    "close_all_requested": False, # <<< FLAG ADDED HERE
    "bot_event_loop": None,       # Stores the asyncio loop for the bot thread
    "telegram_bot_instance": None # Stores the telegram.Bot instance
    # Add more state variables later
}

# Lock to ensure only one thread modifies the state at a time
state_lock = threading.Lock()

# Helper function to get state safely
def get_state(key, default=None):
    """Safely gets a value from the shared bot_state dictionary."""
    # No need to acquire lock for simple dict read if GIL protects it enough?
    # For safety/consistency, especially if dict grows complex, use lock.
    with state_lock:
        return bot_state.get(key, default)

# Helper function to set state safely
def set_state(key, value):
    """Safely sets a value in the shared bot_state dictionary."""
    with state_lock:
        # logger.debug(f"Updating state: {key} = {value}") # Can be noisy
        bot_state[key] = value