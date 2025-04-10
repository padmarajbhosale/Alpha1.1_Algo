import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_engine.mt5_connector import initialize_mt5
from trading_engine.trade_executor import execute_trade

symbol = "EURUSD"

# 1. Connect to MT5
if not initialize_mt5():
    print("❌ MT5 connection failed.")
    exit()

# 2. Mock Features
features = {
    'BOS_Signal': 1,   # BUY
    'ATR': 0.0012      # for SL via ATR
}

# 3. Execute Trade
result = execute_trade(symbol, features)
print("✅ Execution Result:")
print(result)
