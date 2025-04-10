# backtester.py
# UPDATED: To use multi-class regime predictions for simple strategy.

import logging
import pandas as pd
import os
import sys
import datetime
import math
import argparse
from collections import namedtuple
import time
import numpy as np

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
    # <<< MODIFIED: Import new regime predictor function >>>
    from models.predictor import load_model_and_scaler, make_regime_prediction # Use make_regime_prediction
    from risk_management.risk_manager import ( calculate_trade_parameters, # Keep param calc
                                               SL_METHOD, MAX_SPREAD_POINTS, # MIN_CONFIDENCE no longer used directly here
                                               RISK_PERCENT_PER_TRADE, DEFAULT_SL_POINTS,
                                               SL_ATR_MULTIPLIER, DEFAULT_TP_RR_RATIO )
    import MetaTrader5 as mt5
    from trading_engine.mt5_connector import initialize_mt5, shutdown_mt5
except ImportError as e: logger.critical(f"Import failed: {e}", exc_info=True); sys.exit(1)


# --- MT5 Timeframe Mapping ---
TIMEFRAME_MAP = { "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1, }

# --- Mock MT5 Objects for Backtesting ---
MockSymbolInfo = namedtuple("MockSymbolInfo", ["name", "point", "digits", "spread", "trade_contract_size", "trade_tick_value", "trade_tick_size", "currency_profit", "volume_min", "volume_max", "volume_step"])
MockTick = namedtuple("MockTick", ["time", "bid", "ask", "last", "volume", "flags"])
MockAccountInfo = namedtuple("MockAccountInfo", ["login", "balance", "equity", "currency", "leverage"])

# --- Helper Function for Simulated P/L ---
def calculate_simulated_pnl(trade: dict, current_close_price: float, symbol_info: MockSymbolInfo):
    """ Calculates floating P/L for a simulated trade. """
    pnl = 0.0; point = symbol_info.point; tick_value = symbol_info.trade_tick_value; tick_size = symbol_info.trade_tick_size; contract_size = symbol_info.trade_contract_size
    if point is None or point <= 0 or tick_size is None or tick_size <= 0 or contract_size is None or contract_size <= 0 or tick_value is None:
        logger.warning(f"Invalid symbol props for PNL calc: pt={point}, tsz={tick_size}, tval={tick_value}, csz={contract_size}"); return 0.0
    value_per_point_per_lot = (tick_value / tick_size) * point
    price_diff = 0.0
    if trade['type'] == mt5.ORDER_TYPE_BUY: price_diff = current_close_price - trade['entry_price']
    elif trade['type'] == mt5.ORDER_TYPE_SELL: price_diff = trade['entry_price'] - current_close_price
    points_diff = price_diff / point
    pnl = points_diff * value_per_point_per_lot * trade['lots']
    return pnl

# --- Main Backtesting Function Definition ---
def run_backtest(
    symbol: str, timeframe_str: str, start_date_str: str, end_date_str: str,
    data_file_path: str, initial_balance: float = 10000.0,
    simulated_spread_points: int = 20
    ):
    """ Runs a backtest simulation using REGIME PREDICTIONS. """
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("="*10 + f" Starting REGIME Backtest: {symbol} ({timeframe_str}) - Run: {run_timestamp} " + "="*10)
    logger.info(f"Period: {start_date_str} to {end_date_str}, Data: {data_file_path}")
    logger.info(f"Initial Balance: {initial_balance:.2f}, Sim Spread: {simulated_spread_points} points")
    # Log relevant risk params used for execution
    logger.info(f"Execution Risk Settings: SL Method={SL_METHOD}, Risk={RISK_PERCENT_PER_TRADE:.4%}, SL Pts={DEFAULT_SL_POINTS}, SL ATR Mult={SL_ATR_MULTIPLIER}, TP Ratio={DEFAULT_TP_RR_RATIO}")

    # --- Define Sequence Length ---
    try: sequence_length = int(get_config('TRAIN_SEQUENCE_LENGTH', 60))
    except ValueError: logger.error("Invalid TRAIN_SEQUENCE_LENGTH in config."); return
    logger.info(f"Using sequence length for prediction: {sequence_length}")

    # --- Step 1: Load Data ---
    logger.info(f"Loading data from: {data_file_path}"); historical_data = None
    if not os.path.exists(data_file_path): logger.error(f"Data file not found: {data_file_path}"); return
    try: historical_data = pd.read_csv(data_file_path, parse_dates=['time'], index_col='time'); logger.info(f"Loaded {len(historical_data)} records."); required_cols = ['open', 'high', 'low', 'close', 'tick_volume'];
    except Exception as e: logger.exception(f"Error loading CSV: {e}"); return
    if not all(col in historical_data.columns for col in required_cols): logger.error(f"Data missing cols."); return
    for col in required_cols: historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')
    historical_data.dropna(subset=['open', 'high', 'low', 'close'], inplace=True); logger.debug(f"Shape after NaN drop: {historical_data.shape}")
    try: start_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d'); end_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
    except ValueError as date_e: logger.error(f"Invalid date format: {date_e}"); return
    if historical_data.index.tz is not None: start_dt = pd.Timestamp(start_dt).tz_localize(historical_data.index.tz); end_dt = pd.Timestamp(end_dt).tz_localize(historical_data.index.tz)
    try: historical_data = historical_data.loc[start_dt:end_dt]; logger.info(f"Filtered period: {start_dt} to {end_dt}")
    except Exception as filter_e: logger.exception(f"Error filtering data: {filter_e}"); return
    if historical_data.empty: logger.error("No data in range."); return; logger.info(f"Data ready: {len(historical_data)} points.")

    # --- Step 2: Initialize Backtest State ---
    logger.info("Initializing state..."); balance = initial_balance; equity = initial_balance; open_positions = []; trade_history = []; peak_equity = initial_balance; max_drawdown = 0.0; trade_id_counter = 0
    logger.info("Loading REGIME model/scaler..."); model, scaler = load_model_and_scaler();
    if model is None or scaler is None: logger.error("Failed load model/scaler."); return
    logger.info("Regime Model/scaler loaded.")
    # Get static symbol properties
    logger.warning("Using LIVE MT5 for symbol props!"); mt5_initialized = initialize_mt5()
    if not mt5_initialized: logger.error("MT5 connection failed."); return
    live_symbol_info_for_props = mt5.symbol_info(symbol); shutdown_mt5()
    if not live_symbol_info_for_props: logger.error(f"Cannot get live info for {symbol}."); return
    tf_constant = TIMEFRAME_MAP.get(timeframe_str.upper())
    if tf_constant is None: logger.error(f"Invalid timeframe: {timeframe_str}"); return
    min_bars_for_features = 50
    min_total_bars = max(min_bars_for_features, sequence_length) + 10
    logger.info(f"Minimum bars required: ~{min_total_bars}")

    mock_symbol_info = MockSymbolInfo( name=symbol, point=live_symbol_info_for_props.point, digits=live_symbol_info_for_props.digits, spread=simulated_spread_points, trade_contract_size=live_symbol_info_for_props.trade_contract_size, trade_tick_value=live_symbol_info_for_props.trade_tick_value, trade_tick_size=live_symbol_info_for_props.trade_tick_size, currency_profit=live_symbol_info_for_props.currency_profit, volume_min=live_symbol_info_for_props.volume_min, volume_max=live_symbol_info_for_props.volume_max, volume_step=live_symbol_info_for_props.volume_step)
    point = mock_symbol_info.point; digits = mock_symbol_info.digits; tick_value = mock_symbol_info.trade_tick_value; tick_size = mock_symbol_info.trade_tick_size; contract_size = mock_symbol_info.trade_contract_size

    # --- Nested Helper Function to Close Simulated Trades ---
    def close_simulated_trade(pos_to_close, close_price, close_time, reason):
        nonlocal balance, equity
        logger.info(f"Simulating Close: Trade {pos_to_close['id']} ({('BUY' if pos_to_close['type']==mt5.ORDER_TYPE_BUY else 'SELL')}) due to {reason} @ {close_price:.{digits}f} on bar {close_time}")
        pnl = calculate_simulated_pnl(pos_to_close, close_price, mock_symbol_info); sim_cost = pos_to_close.get('cost', 0.0); net_pnl = pnl - sim_cost
        balance += net_pnl; equity = balance
        logger.info(f" --> Closed PnL={pnl:.2f}, Cost={sim_cost:.2f}, Net={net_pnl:.2f}, NewBalance={balance:.2f}")
        closed_trade = pos_to_close.copy(); closed_trade.update({ 'close_time': close_time, 'close_price': close_price, 'pnl': pnl, 'cost': sim_cost, 'net_pnl': net_pnl, 'status': f'CLOSED_{reason}' }); trade_history.append(closed_trade)
        if pos_to_close in open_positions: open_positions.remove(pos_to_close)

    # --- Step 3: Loop Through Historical Data ---
    logger.info(f"Starting simulation loop: {len(historical_data)} bars...")
    total_bars = len(historical_data); iterator = historical_data.iterrows()
    equity_curve = []
    current_atr = np.nan
    predicted_regime = None # Store latest regime prediction

    for index, current_bar in iterator:
        current_time = index; current_bar_number = historical_data.index.get_loc(index) + 1
        current_equity_start_of_bar = balance
        for pos in open_positions: current_equity_start_of_bar += calculate_simulated_pnl(pos, current_bar['close'], mock_symbol_info)
        if current_bar_number == 1 or current_bar_number % 200 == 0 or current_bar_number == total_bars: # Log less frequently
            logger.info(f" Bar {current_bar_number}/{total_bars} ({current_time}). Eq: {current_equity_start_of_bar:.2f}")

        # 3a: Check SL/TP hits FIRST
        for pos in open_positions[:]:
            pos_sl = pos['sl']; pos_tp = pos['tp']; pos_type = pos['type']; trade_closed = False; close_price = None; reason_closed = None
            if pos_type == mt5.ORDER_TYPE_BUY:
                if current_bar['low'] <= pos_sl: close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['high'] >= pos_tp: close_price = pos_tp; reason_closed = "TP"; trade_closed = True
            elif pos_type == mt5.ORDER_TYPE_SELL:
                if current_bar['high'] >= pos_sl: close_price = pos_sl; reason_closed = "SL"; trade_closed = True
                elif current_bar['low'] <= pos_tp: close_price = pos_tp; reason_closed = "TP"; trade_closed = True
            if trade_closed: close_simulated_trade(pos, close_price, current_time, reason_closed)

        # 3b: Get Regime Prediction <<< MODIFIED >>>
        regime_confidence = 0.0
        if current_bar_number >= min_total_bars:
            history_slice = historical_data.iloc[:current_bar_number]
            features_df = calculate_features(history_slice.reset_index())
            if features_df is not None and not features_df.empty:
                predicted_regime, regime_confidence = make_regime_prediction( # Call new function
                    model, scaler, features_df, sequence_length=sequence_length
                )
                if predicted_regime is not None:
                    logger.debug(f" Regime Prediction @ {current_time}: Class={predicted_regime} (0=R, 1=B, 2=S), Conf={regime_confidence:.4f}")
                    if 'ATRr_14' in features_df.columns: current_atr = features_df['ATRr_14'].iloc[-1]
                    else: current_atr = np.nan
                    if pd.isna(current_atr) or current_atr <= 0: current_atr = np.nan
                else: logger.debug(f"Regime prediction failed."); predicted_regime = None
            else: logger.warning(f"Feature calculation failed."); predicted_regime = None
        else: predicted_regime = None

        # 3c: Implement Regime-Based Entry/Exit Logic <<< MODIFIED >>>
        is_position_open = len(open_positions) > 0

        # --- Exit Logic based on Regime ---
        if is_position_open and predicted_regime is not None:
            pos = open_positions[0] # Assuming only one position
            pos_type = pos['type']
            close_reason = None
            # Exit Long if regime is no longer Bullish (i.e., Range or Bearish)
            if pos_type == mt5.ORDER_TYPE_BUY and predicted_regime != 1:
                close_reason = f"Regime->{predicted_regime}"
            # Exit Short if regime is no longer Bearish (i.e., Range or Bullish)
            elif pos_type == mt5.ORDER_TYPE_SELL and predicted_regime != 2:
                close_reason = f"Regime->{predicted_regime}"
            if close_reason:
                close_simulated_trade(pos, current_bar['close'], current_time, close_reason)
                is_position_open = False # Update status

        # --- Entry Logic based on Regime ---
        if not is_position_open and predicted_regime is not None:
            trade_type = None
            entry_condition_met = False
            if predicted_regime == 1: # Enter Long on Bullish
                trade_type = mt5.ORDER_TYPE_BUY
                entry_condition_met = True
                logger.info(f"Regime Entry Condition: BUY Signal @ {current_time} (Regime=1)")
            elif predicted_regime == 2: # Enter Short on Bearish
                trade_type = mt5.ORDER_TYPE_SELL
                entry_condition_met = True
                logger.info(f"Regime Entry Condition: SELL Signal @ {current_time} (Regime=2)")
            # No entry if predicted_regime == 0 (Ranging)

            if entry_condition_met:
                 latest_atr_for_params = current_atr if SL_METHOD == 'ATR' else None
                 if SL_METHOD == 'ATR' and (pd.isna(latest_atr_for_params) or latest_atr_for_params <= 0):
                      logger.warning(f"Cannot calculate {SL_METHOD} based SL/TP: Invalid ATR ({latest_atr_for_params}). Skipping entry.")
                 else:
                      logger.info(f"Calculating trade parameters...")
                      # Use current equity for risk calc? Or balance? Let's use balance for consistency.
                      mock_account_info = MockAccountInfo(login=12345, balance=balance, equity=equity, currency='USD', leverage=100)
                      next_bar_index = current_bar_number
                      if next_bar_index < total_bars:
                           sim_entry_price = historical_data.iloc[next_bar_index]['open']; sim_entry_time = historical_data.index[next_bar_index]
                           trade_params = calculate_trade_parameters(mock_symbol_info, mock_account_info, trade_type, sim_entry_price, latest_atr_for_params)
                           if trade_params:
                                trade_id_counter += 1;
                                value_per_point_per_lot = (tick_value / tick_size) * point if tick_size > 0 and point > 0 and tick_value != 0 else 0.0
                                entry_cost = (simulated_spread_points * value_per_point_per_lot * trade_params['lot_size']) if value_per_point_per_lot > 0 else 0.0
                                new_pos = {'id': trade_id_counter, 'symbol': symbol, 'type': trade_type, 'entry_time': sim_entry_time, 'entry_price': sim_entry_price, 'lots': trade_params['lot_size'], 'sl': trade_params['stop_loss_price'], 'tp': trade_params['take_profit_price'], 'status': 'OPEN', 'cost': entry_cost, 'pnl': -entry_cost }
                                open_positions.append(new_pos); equity -= entry_cost
                                logger.info(f"Simulated OPEN: ID={trade_id_counter}, {('BUY' if trade_type==mt5.ORDER_TYPE_BUY else 'SELL')} {new_pos['lots']} lots @ {sim_entry_price:.{digits}f}, SL={new_pos['sl']:.{digits}f}, TP={new_pos['tp']:.{digits}f}. Cost={entry_cost:.2f}. Equity={equity:.2f}")
                                is_position_open = True # Update status immediately
                           else: logger.warning(f"Parameter calculation failed for entry @ {current_time}.")
                      else: logger.debug("Regime signal on last bar.")


        # 3h: Update Equity, Drawdown, and other stats for logging
        # ... (equity update and saving logic remains the same) ...
        if current_bar_number >= min_total_bars:
            current_equity_end_of_bar = balance; current_floating_pnl = 0.0
            for pos in open_positions: current_floating_pnl += calculate_simulated_pnl(pos, current_bar['close'], mock_symbol_info)
            current_equity_end_of_bar += current_floating_pnl; equity = current_equity_end_of_bar
            peak_equity = max(peak_equity, equity)
            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, current_drawdown)
            total_exposure_value = 0.0
            if contract_size is not None and contract_size > 0:
                 for pos in open_positions: total_exposure_value += pos['lots'] * contract_size * current_bar['close']
            exposure_percentage = (total_exposure_value / equity * 100.0) if equity > 0 else 0.0
            valid_atr = current_atr if not pd.isna(current_atr) else np.nan
            equity_curve.append({
                'time': current_time, 'equity': round(equity, 2),
                'current_drawdown_pct': round(current_drawdown * 100.0, 2),
                'floating_pnl': round(current_floating_pnl, 2),
                'exposure_pct': round(exposure_percentage, 2),
                'atr_14': round(valid_atr, digits if digits and not pd.isna(valid_atr) else 5) # Add rounding for ATR
            })
            if current_bar_number % 1000 == 0: logger.debug(f"End Bar {current_bar_number}: Eq={equity:.2f}, Peak={peak_equity:.2f}, MaxDD={max_drawdown:.2%}")

    # --- End of loop ---

    # Close EOD positions
    logger.info("Closing End-Of-Backtest open positions...");
    for pos in open_positions[:]:
        final_bar = historical_data.iloc[-1]; close_price = final_bar['close']; close_time = final_bar.name
        close_simulated_trade(pos, close_price, close_time, "EOD")

    # --- Step 4: Calculate & Print Performance Metrics ---
    # ... (metrics calculation and printing remains the same) ...
    logger.info("Backtest loop finished. Calculating metrics..."); total_trades = len(trade_history); net_profit_loss = balance - initial_balance
    gross_profit = 0.0; gross_loss = 0.0; winning_trades = 0; losing_trades = 0; largest_win = 0.0; largest_loss = 0.0
    for trade in trade_history:
        pnl = trade.get('net_pnl', 0.0);
        if pnl > 0: winning_trades += 1; gross_profit += pnl; largest_win = max(largest_win, pnl)
        elif pnl < 0: losing_trades += 1; gross_loss += abs(pnl); largest_loss = min(largest_loss, pnl)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    average_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
    average_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
    risk_reward_ratio = abs(average_win / average_loss) if average_loss != 0 else float('inf')
    print("\n" + "="*20 + " Backtest Results " + "="*20); print(f" Period Tested:           {start_date_str} to {end_date_str}"); print(f" Symbol:                  {symbol} ({timeframe_str})"); print(f" Initial Balance:         {initial_balance:,.2f}"); print(f" Final Balance:           {balance:,.2f}"); print(f" Net P/L:                 {net_profit_loss:,.2f}"); print(f" Net P/L %:               {(net_profit_loss / initial_balance) * 100.0:.2f}%"); print(f" Max Drawdown:            {max_drawdown:.2%}"); print("-" * 58); print(f" Total Trades: {total_trades}"); print(f" Wins: {winning_trades} ({win_rate:.2f}%)"); print(f" Losses: {losing_trades}"); print(f" Profit Factor:           {profit_factor:.2f}"); print(f" Avg Win/Loss Ratio:      {risk_reward_ratio:.2f} ({average_win:.2f}/{average_loss:.2f})"); print(f" Largest Win:             {largest_win:.2f}"); print(f" Largest Loss:            {largest_loss:.2f}"); print("="*58)


    # --- Step 5: Save Results to Files ---
    # ... (saving logic remains the same, filenames updated) ...
    results_dir = os.path.join(project_root, 'backtest_results')
    os.makedirs(results_dir, exist_ok=True)
    # A. Save Detailed Trade List
    trades_filename_base = f"trades_{symbol}_{timeframe_str}_{start_date_str}_to_{end_date_str}_REGIME_{run_timestamp}.csv"
    trades_filepath = os.path.join(results_dir, trades_filename_base)
    try:
        if trade_history:
            trades_df = pd.DataFrame(trade_history)
            if 'entry_time' in trades_df.columns: trades_df['entry_time'] = trades_df['entry_time'].astype(str)
            if 'close_time' in trades_df.columns: trades_df['close_time'] = trades_df['close_time'].astype(str)
            trades_df.to_csv(trades_filepath, index=False)
            logger.info(f"Detailed trade list saved to: {trades_filepath}")
        else: logger.info("No trades executed.")
    except Exception as e: logger.exception(f"Error saving detailed trade list: {e}")
    # B. Append Summary Metrics to Log File
    summary_filepath = os.path.join(results_dir, "backtest_summary_log.csv")
    summary_data = {
        'run_timestamp': run_timestamp, 'strategy_type': 'REGIME_DIRECT', 'symbol': symbol, 'timeframe': timeframe_str, # Changed strategy type
        'start_date': start_date_str, 'end_date': end_date_str,
        'min_confidence': 'N/A', 'sl_method': SL_METHOD, 'tp_rr_ratio': DEFAULT_TP_RR_RATIO, 'risk_perc': RISK_PERCENT_PER_TRADE,
        'initial_balance': initial_balance, 'final_balance': balance, 'net_pnl': net_profit_loss,
        'net_pnl_perc': (net_profit_loss / initial_balance) * 100.0 if initial_balance else 0,
        'max_drawdown_perc': max_drawdown * 100.0, 'total_trades': total_trades,
        'winning_trades': winning_trades, 'losing_trades': losing_trades, 'win_rate_perc': win_rate,
        'profit_factor': profit_factor if profit_factor != float('inf') else np.nan,
        'avg_win_loss_ratio': risk_reward_ratio if risk_reward_ratio != float('inf') else np.nan,
        'average_win': average_win, 'average_loss': average_loss,
        'largest_win': largest_win, 'largest_loss': largest_loss
    }
    try:
        summary_df = pd.DataFrame([summary_data])
        file_exists = os.path.exists(summary_filepath)
        summary_df.to_csv(summary_filepath, mode='a', header=not file_exists, index=False, float_format='%.4f')
        logger.info(f"Summary results appended to: {summary_filepath}")
    except Exception as e: logger.exception(f"Error appending summary results: {e}")
    # C. Save Equity Curve Data
    equity_filename_base = f"equity_{symbol}_{timeframe_str}_{start_date_str}_to_{end_date_str}_REGIME_{run_timestamp}.csv"
    equity_filepath = os.path.join(results_dir, equity_filename_base)
    try:
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            equity_df.to_csv(equity_filepath, index=False, float_format='%.4f')
            logger.info(f"Equity curve data saved to: {equity_filepath}")
        else: logger.info("No equity data recorded.")
    except Exception as e: logger.exception(f"Error saving equity curve data: {e}")

    logger.info("="*10 + " Backtest Function Finished " + "="*10)

# --- Main Execution Block (No changes needed below here) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest on historical data.")
    parser.add_argument("-s", "--symbol", required=True, help="Symbol")
    parser.add_argument("-tf", "--timeframe", required=True, help="Timeframe")
    parser.add_argument("-start", "--startdate", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("-end", "--enddate", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("-d", "--datafile", required=True, help="Path to historical data CSV")
    parser.add_argument("-b", "--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("-spread", "--spreadpoints", type=int, default=20, help="Simulated spread")
    args = parser.parse_args()
    data_file = args.datafile
    log_level_cfg = get_config('LOG_LEVEL', 'INFO').upper();
    try: logging.getLogger().setLevel(logging.getLevelName(log_level_cfg)); logger.info(f"Log Level set: {log_level_cfg}")
    except ValueError: logger.warning(f"Invalid LOG_LEVEL '{log_level_cfg}'."); logging.getLogger().setLevel(logging.INFO)
    run_backtest(
        symbol=args.symbol.upper(), timeframe_str=args.timeframe.upper(),
        start_date_str=args.startdate, end_date_str=args.enddate,
        data_file_path=data_file, initial_balance=args.balance,
        simulated_spread_points=args.spreadpoints
    )
# --- End Main Execution Block ---