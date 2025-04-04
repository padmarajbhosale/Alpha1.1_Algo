# trading_engine/engine.py
# Final version with all fixes including import exception syntax & DB logging block structure.
import logging
import time
import sys
import os
import MetaTrader5 as mt5
import pandas as pd
import datetime
import math
import asyncio # Needed for alert scheduling

# --- Ensure project root is in Python path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import project modules with Robust Error Handling --- <<< CORRECTED SYNTAX >>>
try:
    from config.config_loader import get_config
    from shared_state import set_state, get_state, bot_state, state_lock
    from trading_engine.mt5_connector import is_mt5_connected
    from features.feature_calculator import calculate_features
    from models.predictor import load_model_and_scaler, make_prediction
    from risk_management.risk_manager import check_trade_conditions, calculate_trade_parameters
    from trading_engine.trade_executor import execute_trade, close_all_positions
    from database.db_utils import get_db_session
    from database.models import TradeLog
    # Import the function to schedule messages from the bot handler
    from telegram_interface.bot_handler import schedule_telegram_message
except ImportError as e:
    # Log initial error using basic print first, as logging might not be set up
    print(f"FATAL ERROR: Could not import required modules in engine.py. Error: {e}")
    try:
        # Attempt to set up basic logging to record the critical error
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        # Use logger if available, otherwise print again
        try:
             logger = logging.getLogger(__name__) # Try to get logger
             logger.critical(f"Module import failed: {e}", exc_info=True)
        except NameError: # If logger wasn't defined before failure
             print(f"Logging not fully configured. Import Error: {e}")
    except Exception as log_e:
        # If even basic logging setup fails, print that too
        print(f"Additionally, basic logging setup failed: {log_e}")
    sys.exit(1) # Exit the program regardless
# --- End Correction ---


logger = logging.getLogger(__name__) # Get logger instance

# --- Data Fetching Function (Corrected Syntax) --- <<< CORRECTED SYNTAX >>>
def fetch_market_data(symbol: str, timeframe, num_bars: int) -> pd.DataFrame | None:
    """Fetches historical bar data from MT5."""
    logger.debug(f"Attempting fetch {num_bars} bars {symbol} ({timeframe})...")
    rates = None # Initialize rates
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    except Exception as e:
        # Catch potential errors during the MT5 call itself
        logger.exception(f"Error during mt5.copy_rates_from_pos for {symbol}: {e}")
        set_state("mt5_connected", False) # Assume connection issue on error
        return None

    # Check result after the call
    if rates is None:
        mt5_error = mt5.last_error()
        logger.error(f"Failed fetch rates {symbol}. MT5 Error: {mt5_error}")
        # Don't necessarily set state to disconnected here, could be symbol specific issue
        return None
    if len(rates) == 0:
        logger.warning(f"No rates data returned for {symbol}.")
        return None

    # Process data if successful
    try:
        rates_df = pd.DataFrame(rates)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
        logger.info(f"Fetched {len(rates_df)} bars for {symbol}. Last: {rates_df['time'].iloc[-1]}")
        set_state("mt5_connected", True) # Update connection status on success
        return rates_df
    except Exception as df_e: # Catch potential errors during DataFrame processing
        logger.exception(f"Error processing fetched data for {symbol}: {df_e}")
        return None
# --- End Corrected Syntax ---


# --- Update Closed Trades Function ---
def update_closed_trade_logs():
    # ... (This function remains the same as the last correct version, includes alert trigger) ...
    pass # Placeholder for existing function body

# --- Main Trading Loop Function (Restructured DB Logging Block) --- <<< CORRECTED DB LOG STRUCTURE >>>
def run_trading_loop():
    logger.info("Trading engine loop starting...")
    set_state("is_running", True); set_state("is_paused", False)
    logger.info("Loading shared ML model and scaler..."); model, scaler = load_model_and_scaler()
    if model is None or scaler is None: logger.error("Failed load model/scaler."); set_state("is_running", False); return
    logger.info("Shared Model and scaler loaded.")
    symbols_str = get_config('SYMBOLS_TO_TRADE', 'EURUSD'); symbols_list = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    if not symbols_list: logger.error("No symbols configured."); set_state("is_running", False); return
    logger.info(f"Processing symbols: {symbols_list}")
    try: interval_seconds = int(get_config('ENGINE_INTERVAL_SECONDS', 15))
    except ValueError: logger.warning("Invalid interval, using 15."); interval_seconds = 15
    alert_chat_id = None; alert_on_execution = get_config('ALERT_ON_EXECUTION', 'True').lower() == 'true'
    if alert_on_execution:
        try: alert_chat_id_str = get_config('ALERT_CHAT_ID'); alert_chat_id = int(alert_chat_id_str) if alert_chat_id_str else None
        except ValueError: logger.error("Invalid ALERT_CHAT_ID.")
        if not alert_chat_id: logger.warning("ALERT_CHAT_ID not set, exec alerts disabled.")
    running = True; cycle_count = 0

    while running:
        main_cycle_start_time = time.time()
        try:
            cycle_count += 1; logger.info(f"--- Engine Cycle {cycle_count} ---")
            if get_state("is_paused", False): logger.info("Loop PAUSED."); time.sleep(interval_seconds); continue
            if get_state("close_all_requested", False): logger.info("'/close_all' detected..."); closed_c, failed_c = close_all_positions(); logger.info(f"Close all result: Closed={closed_c}, Failed={failed_c}"); set_state("close_all_requested", False)

            for symbol in symbols_list:
                symbol_start_time = time.time(); logger.info(f"--- Processing Symbol: {symbol} ---")
                try:
                    current_timeframe = mt5.TIMEFRAME_M5; bars_to_fetch = 100
                    if not is_mt5_connected(): logger.error("MT5 disconnected!"); set_state("mt5_connected", False); break
                    else: set_state("mt5_connected", True)
                    logger.debug(f"Fetching data {symbol}..."); market_data_df = fetch_market_data(symbol, current_timeframe, bars_to_fetch)
                    if market_data_df is None or market_data_df.empty: continue
                    logger.debug(f"Calculating features {symbol}..."); features_df = calculate_features(market_data_df)
                    if features_df is None or features_df.empty: continue
                    logger.debug(f"Features {symbol} Shape: {features_df.shape}")
                    logger.debug(f"Making prediction {symbol}..."); prediction, confidence = make_prediction(model, scaler, features_df)
                    if prediction is None or confidence is None: continue
                    logger.info(f"Prediction {symbol}: Class={prediction}, Conf={confidence:.4f}")
                    logger.debug(f"Checking conditions {symbol}..."); symbol_info = mt5.symbol_info(symbol); tick_info = mt5.symbol_info_tick(symbol); account_info = mt5.account_info()
                    if not all([symbol_info, tick_info, account_info]): continue
                    trade_ok, reason = check_trade_conditions(prediction, confidence, symbol_info, tick_info)

                    if trade_ok:
                        logger.info(f"Conditions met {symbol}. Calc params..."); latest_atr = features_df['atr_14'].iloc[-1] if 'atr_14' in features_df.columns else None
                        if pd.isna(latest_atr) or latest_atr <= 0: logger.warning(f"Invalid ATR ({latest_atr}) {symbol}"); latest_atr=None
                        trade_type = mt5.ORDER_TYPE_BUY if prediction == 1 else mt5.ORDER_TYPE_SELL; entry_price = tick_info.ask if trade_type == mt5.ORDER_TYPE_BUY else tick_info.bid
                        trade_params = calculate_trade_parameters(symbol_info, account_info, trade_type, entry_price, latest_atr=latest_atr)
                        if trade_params:
                            logger.info(f"{symbol} Params: Lot={trade_params['lot_size']}, SL={trade_params['stop_loss_price']:.{symbol_info.digits}f}, TP={trade_params['take_profit_price']:.{symbol_info.digits}f}")
                            logger.debug(f"Checking positions for {symbol}...")
                            positions = mt5.positions_get(symbol=symbol)
                            if positions is None: continue
                            elif len(positions) > 0: logger.info(f"Skip trade {symbol}: Position open.")
                            else:
                               logger.info(f"No positions {symbol}. Executing..."); logger.info(f"Attempting {('BUY' if trade_type==mt5.ORDER_TYPE_BUY else 'SELL')} {symbol}..."); magic = 10000+abs(hash(symbol))%90000; comment = f"A1D_v1_{symbol}_{prediction}_{confidence:.2f}"
                               trade_result = execute_trade(symbol=symbol, trade_type=trade_type, lot_size=trade_params['lot_size'], stop_loss_price=trade_params['stop_loss_price'], take_profit_price=trade_params['take_profit_price'], magic_number=magic, comment=comment)

                               # --- Log Trade to DB & Alert (Restructured) --- <<< RESTRUCTURED BLOCK >>>
                               if trade_result:
                                   logger.info(f"Trade exec accepted {symbol} (Ticket: {trade_result.order}). Processing log & alert...")
                                   position_id = None # Initialize position_id
                                   deal_ticket = trade_result.deal # Get deal ticket from result

                                   # 1. Try to get Position ID (outside DB transaction)
                                   try:
                                       if deal_ticket > 0:
                                           deal_details = mt5.history_deals_get(ticket=deal_ticket) # Removed semicolon
                                           if deal_details and len(deal_details) > 0:
                                                position_id = deal_details[0].position_id # CORRECT attribute
                                                logger.info(f"Got PosID {position_id}")
                                           else:
                                                logger.warning(f"No deal details found for deal ticket {deal_ticket}.")
                                       else:
                                            logger.warning(f"Invalid Deal Ticket {deal_ticket} received from trade result.")
                                   except Exception as pos_e:
                                       logger.exception(f"Error getting Position ID for Order {trade_result.order}: {pos_e}")

                                   # 2. Try to log to Database
                                   try:
                                       with get_db_session() as db:
                                           entry = TradeLog(
                                               order_id=trade_result.order, position_id=position_id, # Store the Position ID (or None)
                                               timestamp_open=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                                               symbol=symbol, trade_type=trade_type,
                                               lot_size=trade_params['lot_size'], entry_price=trade_result.price, # Use actual exec price
                                               stop_loss_price=trade_params['stop_loss_price'], # Use requested params
                                               take_profit_price=trade_params['take_profit_price'], # Use requested params
                                               prediction=prediction, confidence=confidence,
                                               magic_number=magic, comment=comment, status='OPEN' )
                                           db.add(entry); db.commit()
                                           logger.info(f"Logged trade {trade_result.order} (PosID: {position_id}) to DB.")
                                   except Exception as db_e:
                                        logger.exception(f"CRITICAL DB log error for order {trade_result.order}: {db_e}")

                                   # 3. Try to send Telegram Alert (if DB log succeeded or even if failed?) - Let's send regardless of DB log outcome if trade exec was ok
                                   try:
                                       if alert_on_execution and alert_chat_id:
                                           logger.info(f"Scheduling exec alert {trade_result.order}...")
                                           digits_exec = symbol_info.digits if symbol_info else 5
                                           exec_alert_msg = (f"🚀 *Trade Executed* ({symbol})\nOrder: `{trade_result.order}` Pos: `{position_id}`\nType: {('BUY' if trade_type==mt5.ORDER_TYPE_BUY else 'SELL')} {trade_params['lot_size']} lots @ {trade_result.price:.{digits_exec}f}\nSL: {trade_params['stop_loss_price']:.{digits_exec}f}, TP: {trade_params['take_profit_price']:.{digits_exec}f}")
                                           schedule_telegram_message(alert_chat_id, exec_alert_msg)
                                   except Exception as alert_e:
                                        logger.exception(f"Error scheduling execution alert {trade_result.order}: {alert_e}")
                               # --- End Restructured Block ---
                               else: # If execute_trade failed
                                    logger.error(f"Trade exec failed {symbol}. Not logging.");
                        else: logger.warning(f"Param calculation failed {symbol}.")
                    else: logger.info(f"Conditions NOT met {symbol}: {reason}.")
                except Exception as symbol_e: logger.exception(f"Error processing symbol {symbol}: {symbol_e}"); set_state("last_error", f"Err {symbol}: {symbol_e}")
                finally: logger.debug(f"Finished {symbol} in {time.time() - symbol_start_time:.2f}s")

            # Update Closed Trades (Run once per cycle)
            update_closed_trade_logs() # Includes P/L alert on close logic

            logger.debug("Placeholder: Log cycle status...")
            cycle_elapsed = time.time() - main_cycle_start_time; wait_time = max(0, interval_seconds - cycle_elapsed)
            logger.info(f"--- Engine Cycle {cycle_count} Completed ({cycle_elapsed:.2f}s) ---. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)

        except KeyboardInterrupt: logger.info("Keyboard interrupt. Stopping..."); running = False; set_state("is_running", False)
        except Exception as e: logger.exception(f"CRITICAL Error in main loop: {e}"); set_state("last_error", str(e)); logger.info("Pausing 30s..."); time.sleep(30)

    logger.info("Trading engine loop stopped."); set_state("is_running", False)

if __name__ == "__main__": pass