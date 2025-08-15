# notifier.py
import telegram
import asyncio
import os
import config
from datetime import datetime
from telegram.ext import ApplicationBuilder, Application
from telegram import constants
import logging
import numpy as np

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = config.TELEGRAM_TOKEN
TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID

def initialize_telegram_bot() -> Application:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.strip() == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.critical("TELEGRAM_BOT_TOKEN is not set or is a placeholder in config.py. Alerts will not be sent.")
        return None
    try:
        application = (
            ApplicationBuilder()
            .token(TELEGRAM_BOT_TOKEN)
            .build()
        )
        logger.info("Telegram bot initialized using ApplicationBuilder.")
        return application
    except Exception as e:
        logger.critical(f"Failed to initialize Telegram bot: {e}. Check your bot token in config.py.", exc_info=True)
        return None

async def send_alert(application: Application, message_text: str, reply_to_message_id: int = None):
    if application is None or application.bot is None:
        logger.error("Telegram bot not initialized. Cannot send alert.")
        return None
    if not TELEGRAM_CHAT_ID or str(TELEGRAM_CHAT_ID).strip() == "":
        logger.error("Telegram chat ID is not configured or is empty in config.py. Alerts will not be sent.")
        return None
    try:
        sent_message = await application.bot.send_message(
            chat_id=str(TELEGRAM_CHAT_ID),
            text=message_text,
            parse_mode=constants.ParseMode.HTML,
            reply_to_message_id=reply_to_message_id
        )
        logger.info(f"Telegram message sent: {message_text.splitlines()[0]}...")
        return sent_message.message_id
    except telegram.error.TimedOut as e:
        logger.error(f"Telegram TimedOut: Request timed out. Error: {e}")
    except telegram.error.BadRequest as e:
        logger.error(f"Telegram BadRequest: {e}. Possible issues: incorrect chat ID, bot not started in chat, or invalid HTML/message formatting.")
    except telegram.error.NetworkError as e:
        logger.error(f"Telegram NetworkError: {e}. Check your internet connection or firewall rules.")
    except telegram.error.TelegramError as e:
        logger.error(f"Telegram API Error: {e}. Check your bot token, chat ID, or bot status in chat. Error Type: {type(e).__name__}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending Telegram alert: {e}", exc_info=True)
    return None

async def send_signal_alert(application: Application, trade_obj, chart_pattern_info: str = ""):
    def format_price(price):
        if price is None or (isinstance(price, (float, np.float64)) and np.isnan(price)):
            return "N/A"
        return f"${price:.4f}"

    tp_formatted_list = []
    if trade_obj.tps:
        for i, tp in enumerate(trade_obj.tps):
            if tp is None or (isinstance(tp, (float, np.float64)) and np.isnan(tp)):
                tp_formatted_list.append(f"      - Target {i+1}: <code>N/A</code>")
            else:
                tp_formatted_list.append(f"      - Target {i+1}: <code><b>{format_price(tp)}</b></code>")
    tp_formatted = "\n".join(tp_formatted_list) if tp_formatted_list else "      - None"

    direction_color = '#28a745' if trade_obj.direction.upper() == 'BUY' else '#dc3545'

    message = (
        f"🔔 <b style='color: #007bff;'>NEW TRADE SIGNAL!</b> 🔔\n\n"
        f"📊 Pair: <b><code>{trade_obj.symbol}</code></b>\n"
        f"⬆️⬇️ Direction: <b style='color: {direction_color}'>{trade_obj.direction}</b>\n"
        f"💰 Entry Price: <code>{format_price(trade_obj.entry_price)}</code>\n"
        f"🛑 Stop Loss: <code>{format_price(trade_obj.sl)}</code>\n\n"
        f"🎯 Take Profits:\n{tp_formatted}\n\n"
        f"✨ <i>Trade Safely!</i>"
        
    )
    return await send_alert(application, message)

async def send_tp_alert(application: Application, trade, tp_num, tp_price):
    current_price_for_alert = tp_price
    if not hasattr(trade, 'closed_time') or trade.closed_time is None:
        trade.closed_time = datetime.now()
    all_tps_hit = all(trade.tp_hits)
    status_line = "<b><u>Trade Fully Closed!</u></b>\n" if all_tps_hit else "<u>Partial Profit Unlocked!</u>\n"
    
    def format_price(price):
        if price is None or (isinstance(price, (float, np.float64)) and np.isnan(price)):
            return "N/A"
        return f"${price:.4f}"

    message = (
        f"✅ <b>TAKE PROFIT {tp_num} HIT!</b> ✅\n"
        f"Pair: <b><code>{trade.symbol}</code></b>\n"
        f"Direction: <i>{trade.direction}</i>\n"
        f"Entry Price: <code>{format_price(trade.entry_price)}</code>\n"
        f"TP Price: <code>{format_price(tp_price)}</code>\n"
        f"Closed Price: <code>{format_price(current_price_for_alert)}</code>\n"
        f"Time Closed: <i>{trade.closed_time.strftime('%Y-%m-%d %H:%M:%S')}</i>\n\n"
        f"{status_line}"
        f"🚀💰🚀💰"
    )
    await send_alert(application, message, reply_to_message_id=trade.telegram_message_id)

async def send_final_alert_func(application: Application, trade, final_message: str):
    """Sends the final trade closure alert with a custom message."""
    if trade.telegram_message_id is None:
        logger.error(f"Cannot send final alert for {trade.symbol}. No initial message ID found.")
        return None
    
    await send_alert(
        application, 
        message_text=final_message, 
        reply_to_message_id=trade.telegram_message_id
    )

async def send_error_alert(application: Application, error_message: str):
    message = (
        f"⚠️ <b>ERROR ALERT!</b> ⚠️\n\n"
        f"An unhandled error occurred in the bot:\n"
        f"<code>{error_message}</code>\n\n"
        f"Please check the bot logs for more details."
    )
    await send_alert(application, message)

