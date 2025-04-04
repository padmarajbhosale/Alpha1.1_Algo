# telegram_interface/bot_handler.py
# Fix SyntaxError in run_bot_in_thread
import logging
import sys
import os
import threading
import asyncio
import traceback
import datetime
import MetaTrader5 as mt5

try:
    from telegram import Update, Bot
    from telegram.constants import ParseMode
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters as Filters
    from telegram.error import TelegramError
except ImportError: print("ERROR: python-telegram-bot library not found."); sys.exit(1)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)
try:
    from config.config_loader import get_config
    from shared_state import get_state, set_state, bot_state, state_lock
    from database.db_utils import get_db_session
    from database.models import TradeLog
    from sqlalchemy import func
except ImportError as e: print(f"FATAL ERROR: Import failed: {e}"); sys.exit(1)

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
     handler = logging.StreamHandler(); handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
     logger.addHandler(handler); logger.setLevel(logging.INFO); logger.warning("Basic logging handler added in bot_handler.")

# --- Async Helper to Send Message ---
async def send_telegram_message_async(bot: Bot, chat_id: int, message_text: str):
    # ... (same as before) ...
    if not bot: logger.error("Bot instance None in send_async."); return
    if not chat_id: logger.error("Invalid chat_id in send_async."); return
    try: await bot.send_message(chat_id=chat_id, text=message_text, parse_mode=ParseMode.MARKDOWN); logger.info(f"Msg sent via bot to {chat_id}.")
    except TelegramError as e: logger.error(f"TG API error sending to {chat_id}: {e}")
    except Exception as e: logger.exception(f"Unexpected error send_async to {chat_id}: {e}")

# --- Sync Function to Schedule Sending ---
def schedule_telegram_message(chat_id: int, message_text: str):
    # ... (same as before) ...
    logger.debug(f"schedule_telegram_message called from thread: {threading.current_thread().name}")
    loop = None; bot_instance = None
    with state_lock: loop = bot_state.get("bot_event_loop"); bot_instance = bot_state.get("telegram_bot_instance")
    if not loop: logger.error("Cannot schedule msg: Bot loop not found."); return
    if not bot_instance: logger.error("Cannot schedule msg: Bot instance not found."); return
    if not chat_id: logger.error("Cannot schedule msg: Invalid chat_id."); return
    try: asyncio.run_coroutine_threadsafe(send_telegram_message_async(bot_instance, chat_id, message_text), loop); logger.info(f"Scheduled msg send to {chat_id}.")
    except Exception as e: logger.exception(f"Failed schedule msg send: {e}")

# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (same) ...
    user = update.effective_user; logger.info(f"Handler: /start from {user.username}"); reply_text = f"Hi {user.mention_html()}! Bot active."; await update.message.reply_html(reply_text); await status_command(update, context)
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (same) ...
    logger.info(f"Handler: /help from {update.effective_user.username}"); help_text = "Commands:\n/start\n/help\n/status\n/pause\n/resume\n/daily_report\n/close_all"; await update.message.reply_text(help_text)
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (same as before - uses asyncio.to_thread) ...
    logger.info(f"Handler: /status from {update.effective_user.username}"); is_running = get_state("is_running", False); is_paused = get_state("is_paused", False); mt5_conn = get_state("mt5_connected", False); last_err = get_state("last_error"); status_lines = []
    if is_running: status_lines.append(f"*Engine State:* {'PAUSED' if is_paused else 'RUNNING'}")
    else: status_lines.append("*Engine State:* STOPPED"); status_lines.append(f"*MT5 Connection:* {'CONNECTED' if mt5_conn else 'DISCONNECTED'}");
    balance_str = "N/A"; equity_str = "N/A"; pos_count_str = "N/A"; currency = ""
    if mt5_conn:
        logger.debug("Fetching MT5 info async...");
        try:
            account_info = await asyncio.to_thread(mt5.account_info); positions_total = await asyncio.to_thread(mt5.positions_total)
            if account_info: currency = account_info.currency; balance_str = f"{account_info.balance:.2f}"; equity_str = f"{account_info.equity:.2f}"; pos_count_str = f"{positions_total if positions_total is not None else 'Error'}"; logger.debug("Fetched MT5 info OK.")
            else: logger.warning("account_info() None."); balance_str = "Error(None)"
        except Exception as e: logger.error(f"Exc fetching MT5 info: {e}"); balance_str = "Error(Exc)"
    status_lines.append(f"*Balance:* {balance_str} {currency}"); status_lines.append(f"*Equity:* {equity_str} {currency}"); status_lines.append(f"*Open Positions:* {pos_count_str}");
    if get_state("mt5_connected") and account_info: # Fetch position details only if connected and account_info retrieved
        logger.debug("Fetching position details async..."); positions = None
        try: positions = await asyncio.to_thread(mt5.positions_get)
        except Exception as e: logger.error(f"Exc fetching positions: {e}")
        if positions is not None:
             status_lines.append(f"*Open Positions ({len(positions)}):*")
             if len(positions) > 0:
                  total_floating_pl = 0.0
                  for pos in positions:
                       pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"; digits = mt5.symbol_info(pos.symbol).digits if mt5.symbol_info(pos.symbol) else 5; sl_str = f"SL:{pos.sl:.{digits}f}" if pos.sl > 0 else "SL:N/A"; tp_str = f"TP:{pos.tp:.{digits}f}" if pos.tp > 0 else "TP:N/A"
                       status_lines.append(f"  `{pos.ticket}`: {pos.symbol} {pos_type} {pos.volume} @ {pos.price_open:.{digits}f} ({sl_str} {tp_str}) P/L: {pos.profit:.2f}"); total_floating_pl += pos.profit
                  status_lines.append(f"*Total Floating P/L:* {total_floating_pl:.2f} {currency}")
             else: status_lines.append("  None")
        else: status_lines.append("*Open Positions:* Error fetching")
    if last_err: status_lines.append(f"*Last Error:* {last_err}");
    status_message = "\n".join(status_lines);
    try: logger.debug(f"Attempting /status reply..."); await update.message.reply_text(status_message, parse_mode=ParseMode.MARKDOWN); logger.info("/status reply sent.")
    except Exception as e: logger.exception(f"Failed /status reply: {e}")
async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (same) ...
    logger.info(f"Handler: /pause from {update.effective_user.username}"); set_state("is_paused", True); reply_text = "Engine PAUSED."; await update.message.reply_text(reply_text); logger.info("Engine state set to PAUSED.")
async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (same) ...
     logger.info(f"Handler: /resume from {update.effective_user.username}"); set_state("is_paused", False); reply_text = "Engine RESUMED."; await update.message.reply_text(reply_text); logger.info("Engine state set to RUNNING."); await status_command(update, context)
async def daily_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (same) ...
     logger.info(f"Handler: /daily_report from {update.effective_user.username}"); await update.message.reply_text("Generating report...");
     try:
         now_utc=datetime.datetime.now(datetime.timezone.utc); today_start_utc=now_utc.replace(hour=0,minute=0); today_start_naive=today_start_utc.replace(tzinfo=None); tomorrow_start_naive=(today_start_utc + datetime.timedelta(days=1)).replace(tzinfo=None); total_pnl=0.0; trade_count=0; wins=0
         with get_db_session() as db: closed = db.query(TradeLog).filter(TradeLog.status == 'CLOSED', TradeLog.timestamp_close >= today_start_naive, TradeLog.timestamp_close < tomorrow_start_naive).all(); trade_count = len(closed)
         if trade_count > 0:
             for t in closed: pnl = (t.profit or 0.0)+(t.commission or 0.0)+(t.swap or 0.0); total_pnl += pnl;
             if t.profit > 0: wins += 1
         win_rate = (wins/trade_count*100) if trade_count > 0 else 0; report=[f"*Daily Report (UTC)*", f"Closed: {trade_count}"];
         if trade_count > 0: report.extend([f"Wins: {wins} ({win_rate:.1f}%)", f"Net P/L: ${total_pnl:.2f}"]);
         else: report.append("No trades closed today."); await update.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN); logger.info(f"Sent daily report ({trade_count} trades).")
     except Exception as e: logger.exception(f"Failed /daily_report: {e}"); await update.message.reply_text("Failed report gen.")
async def close_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (same) ...
    user = update.effective_user; logger.info(f"Handler: /close_all from {user.username}"); set_state("close_all_requested", True); reply_text = ("Received /close_all request. Engine will process."); await update.message.reply_text(reply_text); logger.info("Close all flag SET.")
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    # ... (same) ...
    logger.error(f"Exception caught by TG handler: {context.error}", exc_info=context.error); set_state("last_error", f"TG Error: {context.error}")
    # ... (optional reply to user) ...

# --- Main Bot Setup & Polling Function ---
def run_bot_polling():
    token = get_config('TELEGRAM_BOT_TOKEN');
    if not token or token=='YOUR_COPIED_TOKEN_HERE': logger.error("TG token invalid."); return
    logger.info("Setting up Telegram application..."); application = Application.builder().token(token).build()
    with state_lock: bot_state["telegram_bot_instance"] = application.bot; logger.info("Stored telegram bot instance.")
    # Register Handlers
    application.add_handler(CommandHandler("start", start_command)); application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command)); application.add_handler(CommandHandler("pause", pause_command))
    application.add_handler(CommandHandler("resume", resume_command)); application.add_handler(CommandHandler("daily_report", daily_report_command))
    application.add_handler(CommandHandler("close_all", close_all_command))
    application.add_error_handler(error_handler)
    logger.info("Starting Telegram polling..."); application.run_polling(allowed_updates=Update.ALL_TYPES); logger.info("Telegram polling stopped.")

# --- Thread Target Function (Corrected Syntax) --- <<< CORRECTED >>>
def run_bot_in_thread():
    """ Sets up asyncio loop, runs polling, stores loop/bot in shared state. """
    logger.info("Telegram bot thread started. Setting up asyncio loop...")
    loop = None
    try:
        # Create and set a new event loop for this specific thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Asyncio loop created/set for Telegram thread.")

        # Store event loop in shared state BEFORE starting the bot polling,
        # so other threads can potentially schedule tasks on it.
        with state_lock:
             bot_state["bot_event_loop"] = loop
        logger.info("Stored bot event loop in shared state.")

        # Now run the original blocking function that sets up handlers,
        # stores the bot instance, and starts polling.
        run_bot_polling()

    except Exception as e:
        logger.exception(f"Exception in TG thread target: {e}")
        set_state("last_error", f"TG Thread Exception: {e}")
    finally:
        logger.info("Telegram bot thread finishing.")
        # Optional: Clean up loop if needed, though daemon thread might just exit
        # if loop and not loop.is_closed():
        #     loop.close()
        #     logger.info("Asyncio event loop closed.")

# --- Standalone Test Block ---
if __name__ == '__main__':
    pass # Minimal