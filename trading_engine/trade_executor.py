# trading_engine/trade_executor.py
import logging
import MetaTrader5 as mt5
import time
import sys
import os # Only needed if using standalone test block with path adjustments

logger = logging.getLogger(__name__)
# Basic handler if none configured by main.py/caller
if not logging.getLogger().hasHandlers():
     handler = logging.StreamHandler(); handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
     logger.addHandler(handler); logger.setLevel(logging.INFO); logger.warning("Basic logging handler added in trade_executor.")

def execute_trade(
    symbol: str, trade_type: int, lot_size: float,
    stop_loss_price: float, take_profit_price: float,
    magic_number: int, comment: str = "Alpha1Data_Trade"
    ) -> mt5.OrderSendResult | None:
    """Sends a market order request to MetaTrader 5 to OPEN a position."""
    # ... (execute_trade function remains the same - uses FOK for opening) ...
    logger.info(f"Attempting send {('BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL')} order: {symbol}, Lots={lot_size}...")
    tick = mt5.symbol_info_tick(symbol);
    if tick is None: logger.error(f"Tick info unavailable {symbol}."); return None
    price = tick.ask if trade_type == mt5.ORDER_TYPE_BUY else tick.bid; deviation_points = 20
    stop_loss_price = float(stop_loss_price); take_profit_price = float(take_profit_price); lot_size = float(lot_size)
    request = { "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size, "type": trade_type,
                "price": price, "sl": stop_loss_price, "tp": take_profit_price, "deviation": deviation_points,
                "magic": magic_number, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK, # Opening uses FOK
               }
    logger.debug(f"Sending order request: {request}")
    try:
        if not mt5.terminal_info(): logger.error("MT5 connection lost."); return None
        result = mt5.order_send(request)
    except Exception as e: logger.exception(f"Error during order_send: {e}"); return None
    if result is None: logger.error(f"order_send failed, returned None. Last error: {mt5.last_error()}"); return None
    logger.debug(f"Order send raw result: {result}"); logger.info(f"Order send attempt result: Retcode={result.retcode}, Comment={result.comment}, OrderID={result.order}")
    if result.retcode == mt5.TRADE_RETCODE_DONE: logger.info(f"Order accepted! Ticket: {result.order}"); return result
    else: logger.error(f"Order FAILED. Retcode: {result.retcode} - {result.comment}"); logger.error(f"MT5 Last Error: {mt5.last_error()}"); return None


# --- Function to Close Positions (Using FOK Filling) --- <<< CONFIRM FILLING TYPE IS FOK >>>
def close_all_positions(symbol: str = None, magic: int = None) -> tuple[int, int]:
    """Attempts to close open positions, using FOK filling mode."""
    filter_msg = f"symbol={'ANY' if symbol is None else symbol}, magic={'ANY' if magic is None else magic}"
    logger.info(f"Attempting close positions matching filter: {filter_msg}")
    closed_count = 0; failed_count = 0
    try:
        # Get positions based on filter
        if symbol and magic: positions = mt5.positions_get(symbol=symbol, magic=magic)
        elif symbol: positions = mt5.positions_get(symbol=symbol)
        elif magic: positions = mt5.positions_get(magic=magic)
        else: positions = mt5.positions_get()

        if positions is None: logger.error(f"Failed get positions. Error: {mt5.last_error()}"); return 0, 0
        if len(positions) == 0: logger.info(f"No open positions matching filter ({filter_msg})."); return 0, 0
        logger.info(f"Found {len(positions)} positions matching filter ({filter_msg}). Attempting closure...")

        for position in positions:
            position_symbol = position.symbol; position_ticket = position.ticket; position_volume = position.volume
            position_type = position.type; position_magic = position.magic
            if symbol is not None and position_symbol != symbol: continue
            if magic is not None and position_magic != magic: continue

            close_trade_type = mt5.ORDER_TYPE_SELL if position_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_symbol_info = mt5.symbol_info(position_symbol); close_tick_info = mt5.symbol_info_tick(position_symbol)
            if close_symbol_info is None or close_tick_info is None: logger.error(f"No info for {position_symbol}. Cannot close Pos {position_ticket}."); failed_count += 1; continue

            close_price = close_tick_info.bid if close_trade_type == mt5.ORDER_TYPE_BUY else close_tick_info.ask
            deviation = 20

            close_request = {
                "action": mt5.TRADE_ACTION_DEAL, "position": position_ticket, "symbol": position_symbol,
                "volume": position_volume, "type": close_trade_type, "price": close_price,
                "deviation": deviation, "magic": position_magic, # Can use a specific closing magic# if needed
                "comment": "Closed by /close_all", "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK, # <<< Use FOK (Fill Or Kill) instead of IOC
            }
            logger.info(f"Sending close request for Pos {position_ticket} ({position_symbol} {('BUY' if position_type==0 else 'SELL')} {position_volume})...")
            logger.debug(f"Close request details: {close_request}")
            try: result = mt5.order_send(close_request)
            except Exception as e_close: logger.exception(f"Error sending close req {position_ticket}: {e_close}"); failed_count += 1; continue

            if result is None: logger.error(f"Close request Pos {position_ticket} failed (None). MT5 Err: {mt5.last_error()}"); failed_count += 1
            elif result.retcode == mt5.TRADE_RETCODE_DONE: logger.info(f"Close req Pos {position_ticket} accepted. Result: {result.comment}, Deal: {result.deal}"); closed_count += 1
            else: logger.error(f"Close req Pos {position_ticket} FAILED. Retcode: {result.retcode} - {result.comment}. MT5 Err: {mt5.last_error()}"); failed_count += 1
            time.sleep(0.2) # Small delay
    except Exception as e: logger.exception(f"Error during close_all_positions: {e}"); return closed_count, failed_count
    logger.info(f"Finished close_all_positions. Accepted: {closed_count}, Failed: {failed_count}")
    return closed_count, failed_count

# --- Standalone Test Block ---
if __name__ == '__main__':
    # ... (Same as before) ...
    pass