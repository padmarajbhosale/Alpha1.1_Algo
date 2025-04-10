import time
import logging
from dotenv import load_dotenv
from trading_engine.mt5_connector import connect_to_mt5
from trading_engine.trade_executor import execute_trade
from models.predictor import predict_meta_model
from shared_state import get_latest_features, get_symbol_list, get_config
from utils.telegram_utils import send_telegram_alert

# --- Load Environment Config ---
load_dotenv()
logger = logging.getLogger(__name__)
config = get_config()

# --- Initialize MT5 ---
if not connect_to_mt5():
    logger.error("❌ MT5 initialization failed.")
    exit()

# --- Load Symbols ---
symbols = get_symbol_list()
logger.info(f"📊 Loaded symbols: {symbols}")

# --- Main Trading Loop ---
def run_trading_loop():
    logger.info("🚀 Starting live trading loop...")

    while True:
        for symbol in symbols:
            try:
                logger.info(f"\n🧠 Checking {symbol}")

                # --- 1. Get latest features from shared_state / pipeline ---
                features = get_latest_features(symbol)
                if not features:
                    logger.warning(f"No features for {symbol}, skipping.")
                    continue

                # --- 2. Predict using Meta Model ---
                should_trade, confidence = predict_meta_model(features, threshold=0.6)

                if not should_trade:
                    logger.info(f"🛑 Meta model blocked trade for {symbol} | Confidence: {confidence:.2f}")
                    continue

                logger.info(f"✅ Meta model APPROVED trade for {symbol} | Confidence: {confidence:.2f}")

                # --- 3. Telegram Alert ---
                send_telegram_alert(f"✅ TRADE SIGNAL: {symbol} | Confidence: {confidence:.2f}")

                # --- 4. Execute Trade (MT5) ---
                trade_result = execute_trade(symbol, features)
                logger.info(f"📈 Executed: {trade_result}")

            except Exception as e:
                logger.error(f"⚠️ Error in trading loop for {symbol}: {e}")

        time.sleep(60)  # Run every minute (or adjust as needed)
