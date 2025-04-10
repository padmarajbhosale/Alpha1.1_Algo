import MetaTrader5 as mt5
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Config ---
TP_PERCENT = float(os.getenv("TP_PERCENT", 0.007))
SL_PERCENT = float(os.getenv("SL_PERCENT", 0.005))
RISK_PERCENT = float(os.getenv("RISK_PERCENT_PER_TRADE", 0.0025))
DEFAULT_VOLUME = 0.1
POINT_SIZE = float(os.getenv("POINT_SIZE", 0.00001))
SL_METHOD = os.getenv("SL_METHOD", "ATR")
SL_ATR_MULTIPLIER = float(os.getenv("SL_ATR_MULTIPLIER", 2.0))

def execute_trade(symbol, features):
    try:
        # --- Direction ---
        trade_type = mt5.ORDER_TYPE_BUY if features['BOS_Signal'] == 1 else mt5.ORDER_TYPE_SELL

        # --- Price and Point ---
        tick = mt5.symbol_info_tick(symbol)
        point = mt5.symbol_info(symbol).point
        ask = tick.ask
        bid = tick.bid
        entry_price = ask if trade_type == mt5.ORDER_TYPE_BUY else bid

        # --- Calculate SL/TP ---
        sl_points = SL_ATR_MULTIPLIER * features.get('ATR', 10) / POINT_SIZE if SL_METHOD == "ATR" else SL_PERCENT / POINT_SIZE
        tp_points = TP_PERCENT / POINT_SIZE
        volume = DEFAULT_VOLUME  # Could be dynamic based on risk and balance

        if trade_type == mt5.ORDER_TYPE_BUY:
            sl = entry_price - sl_points * point
            tp = entry_price + tp_points * point
        else:
            sl = entry_price + sl_points * point
            tp = entry_price - tp_points * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
            "magic": 777999
        }

        logger.info(f"📤 Sending order: {request}")
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Order failed: {result.retcode}")
            return {"success": False, "error": result.comment, "code": result.retcode}

        logger.info(f"✅ Order executed: {result}")
        return {
            "success": True,
            "symbol": symbol,
            "type": trade_type,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "volume": volume,
            "ticket": result.order,
        }

    except Exception as e:
        logger.exception(f"⚠️ Exception during trade execution: {e}")
        return {"success": False, "error": str(e)}
