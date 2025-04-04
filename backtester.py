# backtester.py
# Final correct version including TIMEFRAME_MAP and argparse setup.
import logging
import pandas as pd
import os
import sys
import datetime
import math
import argparse
from collections import namedtuple

# --- Setup Path and Logging ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)
logger = logging.getLogger(__name__)
try: from utils.logging_config import setup_logging; setup_logging()
except Exception as log_e: print(f"Logging setup error: {log_e}. Basic config."); logging.basicConfig(level=logging.INFO)

# --- Import project components ---
try:
    from config.config_loader import get_config
    from features.feature_calculator import calculate_features
    from models.predictor import load_model_and_scaler, make_prediction
    from risk_management.risk_manager import ( check_trade_conditions, calculate_trade_parameters,
                                               SL_METHOD, MIN_CONFIDENCE, MAX_SPREAD_POINTS,
                                               RISK_PERCENT_PER_TRADE, DEFAULT_SL_POINTS,
                                               SL_ATR_MULTIPLIER, DEFAULT_TP_RR_RATIO )
    import MetaTrader5 as mt5
    from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5 # Needed for symbol_info fallback
except ImportError as e: logger.critical(f"Import failed: {e}", exc_info=True); sys.exit(1)


# --- MT5 Timeframe Mapping --- <<< REQUIRED DICTIONARY >>>
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}
# --- End Added Block ---


# --- Mock MT5 Objects for Backtesting ---
MockSymbolInfo = namedtuple("MockSymbolInfo", ["name", "point", "digits", "spread", "trade_contract_size", "trade_tick_value", "trade_tick_size", "currency_profit", "volume_min", "volume_max", "volume_step"])
MockTick = namedtuple("MockTick", ["time", "bid", "ask", "last", "volume", "flags"])
MockAccountInfo = namedtuple("MockAccountInfo", ["login", "balance", "equity", "currency", "leverage"])

# --- Helper Function for Simulated P/L ---
def calculate_simulated_pnl(trade: dict, current_close_price: float, symbol_info: MockSymbolInfo):
    """ Calculates floating P/L for a simulated trade in account currency (simplified). """
    pnl = 0.0; point = symbol_info.point; tick_value = symbol_info.trade_tick_value; tick_size = symbol_info.trade_tick_size; contract_size = symbol_info.trade_contract_size
    if not all([isinstance(x, (int, float)) and x > 0 for x in [point, tick_size, contract_size]]) or tick_value == 0: logger.warning(f"Invalid props PNL calc: {symbol_info}"); return 0.0
    value_per_point_per_lot = (tick_value / tick_size) * point; price_diff = 0.0
    if trade['type'] == mt5.ORDER_TYPE_BUY: price_diff = current_close_price - trade['entry_price']
    elif trade['type'] == mt5.ORDER_TYPE_SELL: price_diff = trade['entry_price'] - current_close_price
    points_diff = price_diff / point; pnl = points_diff * value_per_point_per_lot * trade['lots']
    if symbol_info.currency_profit != "USD": logger.warning(f"PNL calc assumes profit ccy is USD for {symbol_info.name}.")
    return pnl

# --- Main Backtesting Function Definition ---
def run_backtest(
    symbol: str, timeframe_str: str, start_date_str: str, end_date_str: str,
    data_file_path: str, initial_balance: float = 10000.0,
    simulated_spread_points: int = 20
    ):
    """ Runs a backtest simulation using historical data. """
    logger.info("="*10 + f" Starting Backtest: {symbol} ({timeframe_str}) " + "="*10)
    logger.info(f"Period: {start_date_str} to {end_date_str}, Data: {data_file_path}")
    logger.info(f"Initial Balance: {initial_balance:.2f}, Sim Spread: {simulated_spread_points} points")
    logger.info(f"Risk Settings: SL Method={SL_METHOD}, Min Conf={MIN_CONFIDENCE}, Max Spread={MAX_SPREAD_POINTS}, Risk={RISK_PERCENT_PER_TRADE:.2%}, SL Pts={DEFAULT_SL_POINTS}, SL ATR Mult={SL_ATR_MULTIPLIER}, TP Ratio={DEFAULT_TP_RR_RATIO}")

    # Step 1: Load Data
    logger.info(f"Loading data from: {data_file_path}"); historical_data = None
    if not os.path.exists(data_file_path): logger.error(f"Data file not found: {data_file_path}"); return
    try: historical_data = pd.read_csv(data_file_path, parse_dates=['time'], index_col='time'); logger.info(f"Loaded {len(historical_data)} records."); required_cols = ['open', 'high', 'low', 'close', 'tick_volume'];
    except Exception as e: logger.exception(f"Error loading initial data: {e}"); return
    if not all(col in historical_data.columns for col in required_cols): logger.error(f"Data missing cols. Found: {historical_data.columns.tolist()}"); return
    for col in required_cols: historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')
    historical_data.dropna(subset=['open', 'high', 'low', 'close'], inplace=True); logger.debug(f"Shape after NaN drop: {historical_data.shape}")
    try: start_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d'); end_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59);
    except ValueError as date_e: logger.error(f"Invalid date format: {date_e}"); return
    if historical_data.index.tz is not None: start_dt = pd.Timestamp(start_dt).tz_localize(historical_data.index.tz); end_dt = pd.Timestamp(end_dt).tz_localize(historical_data.index.tz)
    try: historical_data = historical_data.loc[start_dt:end_dt]; logger.info(f"Filtered period: {start_dt} to {end_dt}")
    except Exception as filter_e: logger.exception(f"Error filtering data: {filter_e}"); return
    if historical_data.empty: logger.error("No data in range."); return; logger.info(f"Data ready: {len(historical_data)} points.")

    # Step 2: Initialize Backtest State
    logger.info("Initializing state..."); balance = initial_balance; equity = initial_balance; open_positions = []; trade_history = []; peak_equity = initial_balance; max_drawdown = 0.0; trade_id_counter = 0
    logger.info("Loading model/scaler..."); model, scaler = load_model_and_scaler();
    if model is None or scaler is None: logger.error("Failed load model/scaler."); return
    logger.info("Model/scaler loaded.")
    # Get static symbol properties
    logger.warning("Using LIVE MT5 connection for symbol props! Ensure MT5 running."); mt5_initialized = initialize_mt5()
    if not mt5_initialized: logger.error("MT5 connection failed."); return
    live_symbol_info_for_props = mt5.symbol_info(symbol); shutdown_mt5()
    if not live_symbol_info_for_props: logger.error(f"Cannot get live info for {symbol}."); return
    tf_constant = TIMEFRAME_MAP.get(timeframe_str.upper()) # Use the map, ensure upper case
    if tf_constant is None: logger.error(f"Invalid timeframe string: {timeframe_str}. Valid: {list(TIMEFRAME_MAP.keys())}"); return
    min_bars_for_features = 50

    # Step 3: Loop Through Historical Data
    logger.info(f"Starting simulation loop: {len(historical_data)} bars...")
    total_bars = len(historical_data); iterator = historical_data.iterrows()

    for index, current_bar in iterator:
        current_time = index; current_bar_number = historical_data.index.get_loc(index) + 1
        if current_bar_number == 1 or current_bar_number % 1000 == 0 or current_bar_number == total_bars: logger.info(f" Bar {current_bar_number}/{total_bars} ({current_time}). Eq: {equity:.2f}")

        # 3a: Check SL/TP hits
        for pos in open_positions[:]:
            pos_sl = pos['sl']; pos_tp = pos['tp']; pos_type = pos['type']; pos_ticket = pos['id']; digits = live_symbol_info_for_props.digits
            trade_closed = False; close_price = None; reason_closed = None
            if pos_type == mt5.ORDER_TYPE_BUY:
                if current_bar['low'] <= pos_sl: close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['high'] >= pos_tp: close_price = pos_tp; reason_closed = "TP"; trade_closed = True
            elif pos_type == mt5.ORDER_TYPE_SELL:
                if current_bar['high'] >= pos_sl: close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['low'] <= pos_tp: close_price = pos_tp; reason_closed = "TP"; trade_closed = True
            if trade_closed:
                logger.info(f"Sim Close: Trade {pos_ticket} hit {reason_closed} @ {close_price:.{digits}f} on {current_time}")
                pnl = calculate_simulated_pnl(pos, close_price, live_symbol_info_for_props); sim_cost = pos.get('cost', 0.0); net_pnl = pnl - sim_cost
                balance += net_pnl; equity = balance # Equity becomes balance after close
                logger.info(f" --> Closed PnL={pnl:.2f}, Cost={sim_cost:.2f}, Net={net_pnl:.2f}, NewBal={balance:.2f}")
                closed_trade = pos.copy(); closed_trade.update({ 'close_time': current_time, 'close_price': close_price, 'pnl': pnl, 'cost': sim_cost, 'net_pnl': net_pnl, 'status': f'CLOSED_{reason_closed}' }); trade_history.append(closed_trade)
                open_positions.remove(pos)

        # 3b/c/d/e/f/g: Check for new trade entry signal
        if current_bar_number >= min_bars_for_features:
            history_slice = historical_data.iloc[:current_bar_number]; features_df = calculate_features(history_slice)
            if features_df is not None and not features_df.empty:
                latest_features = features_df.iloc[-1:]; prediction, confidence = make_prediction(model, scaler, latest_features)
                if prediction is not None and confidence is not None:
                     logger.debug(f" Prediction {current_time}: Class={prediction}, Conf={confidence:.4f}")
                     mock_symbol_info = MockSymbolInfo( name=symbol, point=live_symbol_info_for_props.point, digits=live_symbol_info_for_props.digits, spread=simulated_spread_points, trade_contract_size=live_symbol_info_for_props.trade_contract_size, trade_tick_value=live_symbol_info_for_props.trade_tick_value, trade_tick_size=live_symbol_info_for_props.trade_tick_size, currency_profit=live_symbol_info_for_props.currency_profit, volume_min=live_symbol_info_for_props.volume_min, volume_max=live_symbol_info_for_props.volume_max, volume_step=live_symbol_info_for_props.volume_step)
                     sim_bid = current_bar['close']; sim_ask = sim_bid + (mock_symbol_info.spread * mock_symbol_info.point)
                     mock_tick_info = MockTick(time=current_time.timestamp(), bid=sim_bid, ask=sim_ask, last=current_bar['close'], volume=current_bar['tick_volume'], flags=6)
                     mock_account_info = MockAccountInfo(login=12345, balance=balance, equity=equity, currency='USD', leverage=100)
                     trade_ok, reason = check_trade_conditions(prediction, confidence, mock_symbol_info, mock_tick_info)
                     is_position_open = len(open_positions) > 0 # Simplistic check

                     if trade_ok and not is_position_open:
                         logger.info(f" Trade condition met {current_time}. Calc params...")
                         latest_atr = features_df['atr_14'].iloc[-1] if 'atr_14' in features_df.columns and not pd.isna(features_df['atr_14'].iloc[-1]) else None
                         trade_type = mt5.ORDER_TYPE_BUY if prediction == 1 else mt5.ORDER_TYPE_SELL
                         next_bar_index = current_bar_number
                         if next_bar_index < total_bars:
                             sim_entry_price = historical_data.iloc[next_bar_index]['open']; sim_entry_time = historical_data.index[next_bar_index]; digits = mock_symbol_info.digits
                             trade_params = calculate_trade_parameters(mock_symbol_info, mock_account_info, trade_type, sim_entry_price, latest_atr)
                             if trade_params:
                                 trade_id_counter += 1; entry_cost = (simulated_spread_points * point) * (tick_value / tick_size * point) * trade_params['lots'] if tick_size > 0 and point > 0 and tick_value != 0 else 0.0; # Check tick_value too
                                 new_pos = {'id': trade_id_counter, 'symbol': symbol, 'type': trade_type, 'entry_time': sim_entry_time, 'entry_price': sim_entry_price, 'lots': trade_params['lot_size'], 'sl': trade_params['stop_loss_price'], 'tp': trade_params['take_profit_price'], 'status': 'OPEN', 'cost': entry_cost, 'pnl': -entry_cost }
                                 open_positions.append(new_pos); equity -= entry_cost # Deduct cost from equity
                                 logger.info(f"Simulated OPEN: ID={trade_id_counter}, {('BUY' if trade_type==mt5.ORDER_TYPE_BUY else 'SELL')} {new_pos['lots']} lots @ {sim_entry_price:.{digits}}, SL={new_pos['sl']:.{digits}f}, TP={new_pos['tp']:.{digits}f}. Cost={entry_cost:.2f}. Equity={equity:.2f}")
                             else: logger.warning("Param calc failed.")
                         else: logger.debug("Last bar.")
                     # else: logger.debug(f"No trade: OK={trade_ok}, Open={is_position_open}")

        # 3h: Update Equity & Drawdown
        current_equity = balance;
        for pos in open_positions: current_equity += calculate_simulated_pnl(pos, current_bar['close'], mock_symbol_info)
        equity = current_equity; peak_equity = max(peak_equity, equity); drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0; max_drawdown = max(max_drawdown, drawdown)
        if current_bar_number % 1000 == 0: logger.debug(f"End Bar {current_bar_number}: Eq={equity:.2f}, Peak={peak_equity:.2f}, MaxDD={max_drawdown:.2%}")

    # End of main loop

    # Close EOD positions
    logger.info("Closing EOD positions...");
    for pos in open_positions[:]: final_bar = historical_data.iloc[-1]; close_price = final_bar['close']; close_time = final_bar.name; pnl = calculate_simulated_pnl(pos, close_price, live_symbol_info_for_props); sim_cost = pos.get('cost', 0.0); net_pnl = pnl - sim_cost; balance += net_pnl; equity = balance; closed_trade = pos.copy(); closed_trade.update({ 'close_time': close_time, 'close_price': close_price, 'pnl': pnl, 'cost': sim_cost, 'net_pnl': net_pnl, 'status': 'CLOSED_EOD' }); trade_history.append(closed_trade); open_positions.remove(pos); logger.info(f" Closed EOD PosID={pos['id']}, NetPnL={net_pnl:.2f}, NewBal={balance:.2f}")

    # Step 4: Calculate & Print Metrics
    logger.info("Calculating metrics..."); total_trades = len(trade_history); net_profit_loss = balance - initial_balance
    gross_profit = 0.0; gross_loss = 0.0; winning_trades = 0; losing_trades = 0; largest_win = 0.0; largest_loss = 0.0
    for trade in trade_history: pnl = trade.get('net_pnl', 0.0); # Use net PnL
    if pnl > 0: winning_trades += 1; gross_profit += pnl; largest_win = max(largest_win, pnl)
    elif pnl < 0: losing_trades += 1; gross_loss += abs(pnl); largest_loss = min(largest_loss, pnl)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    average_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
    average_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
    risk_reward_ratio = abs(average_win / average_loss) if average_loss != 0 else float('inf')
    print("\n" + "="*20 + " Backtest Results " + "="*20); print(f" Period: {start_date_str} to {end_date_str}"); print(f" Symbol: {symbol} ({timeframe_str})"); print(f" Initial Balance: {initial_balance:,.2f}"); print(f" Final Balance: {balance:,.2f}"); print(f" Net P/L: {net_profit_loss:,.2f}"); print(f" Net P/L %: {(net_profit_loss / initial_balance) * 100.0:.2f}%"); print(f" Max Drawdown: {max_drawdown:.2%}"); print("-" * 58); print(f" Total Trades: {total_trades}"); print(f" Wins: {winning_trades} ({win_rate:.2f}%)"); print(f" Losses: {losing_trades}"); print(f" Profit Factor: {profit_factor:.2f}"); print(f" Avg Win/Loss Ratio: {risk_reward_ratio:.2f} ({average_win:.2f}/{average_loss:.2f})"); print(f" Largest Win: {largest_win:.2f}"); print(f" Largest Loss: {largest_loss:.2f}"); print("="*58)

    logger.info("="*10 + " Backtest Function Finished " + "="*10)

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest."); parser.add_argument("-s", "--symbol", required=True); parser.add_argument("-tf", "--timeframe", required=True); parser.add_argument("-start", "--startdate", required=True); parser.add_argument("-end", "--enddate", required=True); parser.add_argument("-d", "--datafile"); parser.add_argument("-b", "--balance", type=float, default=10000.0); parser.add_argument("-dir", "--datadir", default="./data"); args = parser.parse_args()
    data_file = args.datafile;
    if not data_file: filename = f"{args.symbol.upper()}_{args.timeframe.upper()}_{args.startdate}_to_{args.enddate}.csv"; data_file = os.path.join(args.datadir, filename); logger.info(f"Using default data file: {data_file}")
    log_level_cfg = get_config('LOG_LEVEL', 'INFO').upper();
    try: level_to_set = logging.getLevelName(log_level_cfg); logging.getLogger().setLevel(level_to_set); logger.info(f"Log Level: {log_level_cfg}")
    except ValueError: logger.warning(f"Invalid LOG_LEVEL '{log_level_cfg}'."); logging.getLogger().setLevel(logging.INFO)

    run_backtest( symbol=args.symbol.upper(), timeframe_str=args.timeframe.upper(), start_date_str=args.startdate, end_date_str=args.enddate, data_file_path=data_file, initial_balance=args.balance )