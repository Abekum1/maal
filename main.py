import asyncio
import os
import ccxt.pro as ccxt
from datetime import datetime, timedelta
import numpy as np
import logging
from predictor import calculate_indicators, run_strategy, Trade, check_and_update_active_trades


# Import notifier functions
from notifier import (
    initialize_telegram_bot,
    send_signal_alert,
    # send_sl_alert, # This is no longer needed
    send_tp_alert,
    send_error_alert,
    send_alert,
    send_final_alert_func, # <-- IMPORT THE NEW FUNCTION
)

# Import predict_trade and the Trade class from predictor.py
# The Trade class is now defined in predictor.py to avoid circular imports
from predictor import predict_trade, Trade

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Master Active Trades Dictionary (Centralized) ---
active_trades_map = {} 
historical_trades = []

SYMBOLS = [
    # Top 20 by Market Cap (Highly Available)
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT',
    'DOGE/USDT:USDT', 'DOT/USDT:USDT', 'LTC/USDT:USDT', 'AVAX/USDT:USDT',
    'LINK/USDT:USDT', 'ATOM/USDT:USDT', 'UNI/USDT:USDT', 'XLM/USDT:USDT', 'FIL/USDT:USDT', 'ICP/USDT:USDT',
    'THETA/USDT:USDT', 'FTM/USDT:USDT',

    # Major Layer 1 & Layer 2 -- High Liquidity
    'ALGO/USDT:USDT', 'NEAR/USDT:USDT', 'FLOW/USDT:USDT', 'EGLD/USDT:USDT', 'ETC/USDT:USDT', 'KAS/USDT:USDT', 
    'APT/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'IMX/USDT:USDT', 'MINA/USDT:USDT', 'SUI/USDT:USDT',
    'SEI/USDT:USDT', 'INJ/USDT:USDT', 'TIA/USDT:USDT', 'KAVA/USDT:USDT',

    # DeFi Tokens
    'AAVE/USDT:USDT', 'COMP/USDT:USDT', 'MKR/USDT:USDT', 'SNX/USDT:USDT', 'CRV/USDT:USDT', 'LDO/USDT:USDT',
    'FXS/USDT:USDT', 'DYDX/USDT:USDT', 'GMX/USDT:USDT', 'RDNT/USDT:USDT', 'BAL/USDT:USDT', 'YFI/USDT:USDT',
    'SUSHI/USDT:USDT', 'CAKE/USDT:USDT', 'JOE/USDT:USDT',

    # Gaming/Metaverse
    'SAND/USDT:USDT', 'MANA/USDT:USDT', 'APE/USDT:USDT', 'GALA/USDT:USDT', 'ENJ/USDT:USDT', 'ILV/USDT:USDT',
    'MAGIC/USDT:USDT', 'YGG/USDT:USDT',

    # AI & Big Data
    'FET/USDT:USDT', 'AGIX/USDT:USDT', 'OCEAN/USDT:USDT', 'NMR/USDT:USDT', 'RLC/USDT:USDT',
    
    # Privacy & Scaling
    'XMR/USDT:USDT', 'ZEC/USDT:USDT', 'DASH/USDT:USDT', 'ZEN/USDT:USDT', 

    # Oracles
    'TRB/USDT:USDT', 'BAND/USDT:USDT', 'UMA/USDT:USDT', 'API3/USDT:USDT',

    # Real World Assets (RWA)
    'ONDO/USDT:USDT', 'TOKEN/USDT:USDT', 
    
    # Other High-Liquidity
    'WAVES/USDT:USDT', 'NEO/USDT:USDT', 'IOTA/USDT:USDT', 'ANKR/USDT:USDT', '1INCH/USDT:USDT', 
    'AR/USDT:USDT', 'GMT/USDT:USDT', 'STX/USDT:USDT', 'PERP/USDT:USDT', 'DUSK/USDT:USDT',
    'ACH/USDT:USDT', 'HIGH/USDT:USDT', 'CELR/USDT:USDT', 'CKB/USDT:USDT',
    
    # User Added Symbols
    'TAO/USDT:USDT', 'OM/USDT:USDT', 'USUAL/USDT:USDT', 'TRX/USDT:USDT', 'TON/USDT:USDT', 'OMNI/USDT:USDT',
    'HBAR/USDT:USDT', 'ROSE/USDT:USDT', 'MANTA/USDT:USDT', 'CROSS/USDT:USDT', 'NEIROETH/USDT:USDT',
    'ORDI/USDT:USDT', 'RENDER/USDT:USDT', 'ARKM/USDT:USDT', 'BCH/USDT:USDT', 'HOT/USDT:USDT', 'RARE/USDT:USDT',
    'SOON/USDT:USDT', 'ZORA/USDT:USDT', 'PUMP/USDT:USDT', 'PROVE/USDT:USDT', 'PENGU/USDT:USDT', 'ZRO/USDT:USDT',
    'MYX/USDT:USDT', 'ZKJ/USDT:USDT', 'BIO/USDT:USDT', 'HYPE/USDT:USDT', 'FUN/USDT:USDT', 'WLD/USDT:USDT',
    'MYRO/USDT:USDT', 'ENS/USDT:USDT', 'VIRTUAL/USDT:USDT', 'NEIRO/USDT:USDT', 'OG/USDT:USDT', 'CFX/USDT:USDT',
    'IP/USDT:USDT', 'PNUT/USDT:USDT', 'ARC/USDT:USDT', 'SPK/USDT:USDT', 'STG/USDT:USDT', 'SAHARA/USDT:USDT',
    'JUP/USDT:USDT', 'LISTA/USDT:USDT', 'PLAY/USDT:USDT', 'SPX/USDT:USDT', 'IN/USDT:USDT', 'ASR/USDT:USDT',
    'TST/USDT:USDT', 'VINE/USDT:USDT', 'HYPER/USDT:USDT', 'PEOPLE/USDT:USDT', 'ALL/USDT:USDT', 'RESOLV/USDT:USDT',
    'ACT/USDT:USDT', 'BOME/USDT:USDT', 'MAV/USDT:USDT', 'VET/USDT:USDT', 'COW/USDT:USDT', 'STRK/USDT:USDT',
    'KAITO/USDT:USDT', 'JASMY/USDT:USDT', 'TAG/USDT:USDT', 'IO/USDT:USDT', 'BSV/USDT:USDT', 'KERNEL/USDT:USDT',
    'SAGA/USDT:USDT', 'DIA/USDT:USDT'
]


CONFIDENCE_THRESHOLD = 100
LOOP_INTERVAL_SECONDS = 300 # 5 minutes


async def generate_daily_report(application):
    now = datetime.now()
    twenty_four_hours_ago = now - timedelta(hours=24)

    recent_trades_data = []
    processed_trade_ids = set()

    for symbol, trade in active_trades_map.items():
        if trade.entry_time >= twenty_four_hours_ago:
            recent_trades_data.append(trade)
            processed_trade_ids.add(id(trade))
    
    for trade in historical_trades:
        if id(trade) not in processed_trade_ids and \
           (trade.entry_time >= twenty_four_hours_ago or \
           (trade.closed_time and trade.closed_time >= twenty_four_hours_ago)):
            recent_trades_data.append(trade)
            processed_trade_ids.add(id(trade))

    total_signals_24h = len(recent_trades_data)
    successful_trades_24h = 0
    failed_trades_24h = 0
    partial_profit_trades_24h = 0

    detailed_trade_outcomes = []

    for trade in recent_trades_data:
        if trade.status == 'CLOSED_TP_FULL':
            successful_trades_24h += 1
            detailed_trade_outcomes.append(
                f"{trade.symbol} ------Full TP------ <b style='color: #28a745;'>🟢 </b>"
            )
        # We now have a new status to check
        elif trade.status == 'CLOSED_SL' or trade.status == 'CLOSED_SL_AFTER_TP':
            failed_trades_24h += 1
            detailed_trade_outcomes.append(
                f"{trade.symbol} ------Stop loss------ <b style='color: #dc3545;'>🔴 </b>"
            )
        elif trade.status == 'CLOSED_TP_PARTIAL':
            partial_profit_trades_24h += 1
            detailed_trade_outcomes.append(
                f"{trade.symbol} ------Partial TP------ <b style='color: #ffc107;'>🟢</b>"
            )

    total_closed_trades = successful_trades_24h + failed_trades_24h + partial_profit_trades_24h 
    success_rate = 0
    if total_closed_trades > 0:
        success_rate = ((successful_trades_24h + partial_profit_trades_24h ) / total_closed_trades) * 100

    report_message = (
        f"📊 24-Hour Trading Report ({now.strftime('%Y-%m-%d %H:%M')}) 📊\n\n"
        f"📈 Signals Generated (last 24h): <b>{total_signals_24h}</b>\n\n"
        f"🟢 Successful Trades (TP Hit): <b style='color: #28a745;'>{partial_profit_trades_24h + successful_trades_24h }</b>\n"
        f"🔴 Failed Trades (SL Hit): <b style='color: #dc3545;'>{failed_trades_24h}</b>\n"
        f"🎯 Success Rate (Closed Trades): <b>{success_rate:.2f}%</b>"
    )

    report_message += "\n\n"
    if detailed_trade_outcomes:
        report_message += "📋 Individual Trade Outcomes:\n"
        for outcome_string in detailed_trade_outcomes:
            report_message += f"   {outcome_string}\n"
    else:
        report_message += "📋 Individual Trade Outcomes : None in last 24h.\n"

    await send_alert(application, report_message)
    logger.info("Daily report sent.")

async def main() -> None:
    application = initialize_telegram_bot()
    logger.info("Bot started. Initial active_trades_map state: %s", list(active_trades_map.keys()))

    exchange = ccxt.binanceusdm()
    last_report_time = datetime.now()
    
    try:
        while True:
            current_time = datetime.now()
            if (current_time - last_report_time).total_seconds() >= 24 * 3600:
                await generate_daily_report(application)
                last_report_time = current_time

            logger.info("\n[INFO] Starting bot cycle...")
            
            tasks = []
            for symbol in SYMBOLS:
                tasks.append(
                    predict_trade(
                        exchange=exchange,
                        symbol=symbol, 
                        active_trades_map=active_trades_map, 
                        TradeClass=Trade,
                        confidence_threshold=CONFIDENCE_THRESHOLD,
                        send_signal_alert_func=lambda trade, chart_info: send_signal_alert(application, trade, chart_info),
                        send_sl_alert_func=None, # This is now None, as the logic is handled by send_final_alert_func
                        send_tp_alert_func=lambda trade, tp_num, current_price: send_tp_alert(application, trade, tp_num, current_price),
                        send_final_alert_func=lambda trade, final_message: send_final_alert_func(application, trade, final_message) # Corrected call
                    )
                )
            
            await asyncio.gather(*tasks)

            symbols_to_remove = []
            for symbol, trade in active_trades_map.items():
                if not trade.active: 
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                closed_trade = active_trades_map.pop(symbol)
                if closed_trade.closed_time is None:
                    closed_trade.closed_time = datetime.now() 
                historical_trades.append(closed_trade)
                logger.info("Moved closed trade %s to historical_trades. Status: %s", closed_trade.symbol, closed_trade.status)

            logger.info("Active trades after cycle: %s", list(active_trades_map.keys()))
            logger.info("Total historical trades: %s", len(historical_trades))
            logger.info("[INFO] Cycle completed. Sleeping for %s seconds...", LOOP_INTERVAL_SECONDS)
            await asyncio.sleep(LOOP_INTERVAL_SECONDS) 
    
    except Exception as e:
        logger.error("An unhandled error occurred in the main loop: %s", e, exc_info=True)
        await send_error_alert(application, str(e))
    finally:
        await exchange.close()
        logger.info("Bot stopped and exchange closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped manually.")