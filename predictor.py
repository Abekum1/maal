# predictor.py
# Higher-accuracy signal generation with structure-based SL/TP, FVG-aware retest, and MACD-first pending entries
# - Keeps 100% of your original content and APIs; only adds safe, optional improvements
# - Trigger-first logic (MACD) with soft filters (EMA, ADX, RSI, Volume)
# - NEW: Pending signal state => wait for retest to support/resistance/FVG before entering
# - SL anchored to recent swing structure or FVG edge with ATR buffer (whichever is stronger)
# - TPs as R-multiples; optional BE after TP1; optional ATR trailing
# - Optional signal cancellation if majority of indicators flip against the position
# - Confidence threshold is enforced in main.py (not here)

import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ccxt.pro as ccxt

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ============================================================
# ------------------------------- Tunables -------------------
# ============================================================
TIMEFRAME = "15m"
OHLCV_LIMIT = 300

# Structure
SWING_LOOKBACK = 8            # candles used to anchor SL to last swing
ATR_MULT_SL_BUFFER = 0.25     # small buffer beyond swing (fraction of ATR)
MIN_SL_DIST_PCT = 0.001       # 0.1% minimal SL distance guard

TP_R_MULTIPLES = [1.0, 1.5, 2.0]  # 1R / 1.5R / 2R

# Quality filters
MAX_DISTANCE_TO_LEVEL_ATR = 0.9    # proximity to SR (<= ATR * this)
EMA_EXTENSION_ATR = 0.7            # "don't chase" guard vs ema_short
MIN_ADX = 18.0                     # weak trend filter

# Reversal cancellation (post-entry)
ENABLE_CANCEL_AFTER_ENTRY = True
CANCEL_AFTER_CANDLES = 4           # start evaluating reversals after N candles post entry
REVERSAL_MAJORITY_THRESHOLD = 3    # how many opposing signals needed to cancel

# Optional per-symbol cooldown (main.py can also implement its own)
SYMBOL_COOLDOWN_SECONDS = 0        # set >0 if you want local cooldown in this module

# ============================================================
# --------- NEW: Pending retest & FVG-aware configuration ----
# ============================================================
REQUIRE_RETEST_AFTER_TRIGGER = True
RETEST_ATR_TOLERANCE = 0.5         # price must retest within 0.5 ATR of SR/FVG level
PENDING_RETEST_MAX_CANDLES = 8     # how long (candles) we wait for retest after MACD trigger
ALLOW_WEAK_MACD_ON_RETEST = True   # MACD can soften during retest (don’t discard the setup)

# NEW: FVG detection (very simple ICT-style, last N candles)
FVG_LOOKBACK = 25

# SL improvements
USE_FVG_FOR_SL = True              # anchor SL to nearest FVG edge if stronger than swing
EXTRA_SL_BUFFER_ATR = 0.05         # tiny extra buffer to reduce SL taps on wicks

# TP/SL management
MOVE_TO_BREAKEVEN_AFTER_TP_INDEX = 1   # 1 => after TP1; 0 to disable
ENABLE_ATR_TRAILING = True
ATR_TRAIL_MULT = 1.0
ATR_TRAIL_LOOKBACK = 14

# Signal throttling
MAX_SIGNALS_PER_DAY_PER_SYMBOL = 6     # hard guard to reduce spam; 0 disables

# ============================================================
# ------------------------------- Data + Indicators ----------
# ============================================================
async def fetch_ohlcv_data(exchange, symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.name = symbol
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error while fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or len(df) < 60:
        return None

    # EMAs (fast/slow like MACD)
    df["ema_short"] = EMAIndicator(close=df["close"], window=12).ema_indicator()
    df["ema_long"]  = EMAIndicator(close=df["close"], window=26).ema_indicator()

    # MACD
    macd = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd_line"]   = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # RSI
    df["rsi"] = RSIIndicator(close=df["close"], window=10).rsi()

    # ADX & DI
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"]        = adx.adx()
    df["adx_pos_di"] = adx.adx_pos()
    df["adx_neg_di"] = adx.adx_neg()

    # ATR
    df["atr"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    # Volume SMA
    df["volume_sma"] = df["volume"].rolling(20).mean()

    needed = [
        "ema_short", "ema_long", "macd_line", "macd_signal",
        "macd_hist", "rsi", "adx", "adx_pos_di", "adx_neg_di",
        "atr", "volume", "volume_sma"
    ]
    if any(pd.isna(df[col].iloc[-1]) for col in needed):
        return None
    return df

# ============================================================
# -------------------- Structure-based helpers ---------------
# ============================================================
def _recent_support(df: pd.DataFrame, lookback=SWING_LOOKBACK) -> float:
    return float(np.min(df["low"].iloc[-lookback:]))

def _recent_resistance(df: pd.DataFrame, lookback=SWING_LOOKBACK) -> float:
    return float(np.max(df["high"].iloc[-lookback:]))

def _distance_to_level(price: float, level: float) -> float:
    return abs(float(price) - float(level))

# ============================================================
# -------------------- FVG (Fair Value Gap) helpers ----------
# ============================================================
def _detect_latest_fvg(df: pd.DataFrame, lookback: int = FVG_LOOKBACK):
    """
    Very simple 3-candle FVG definition:
    - Bullish FVG when low[n+1] > high[n-1]; gap zone = (high[n-1], low[n+1])
    - Bearish FVG when high[n+1] < low[n-1]; gap zone = (high[n+1], low[n-1]) with upper<lower semantics handled
    Returns dict or None:
    {
        "type": "bullish" | "bearish",
        "lower": float,   # min price of gap
        "upper": float,   # max price of gap
        "index": int      # index of the middle candle (n)
    }
    """
    if len(df) < 5:
        return None

    highs = df["high"].values
    lows  = df["low"].values
    end = len(df) - 2  # ensure n+1 exists
    start = max(2, len(df) - lookback)

    latest = None
    for n in range(start, end):
        hi_prev = highs[n - 1]
        lo_prev = lows[n - 1]
        hi_next = highs[n + 1]
        lo_next = lows[n + 1]

        # Bullish gap: lo_next > hi_prev
        if lo_next > hi_prev:
            lower = hi_prev
            upper = lo_next
            latest = {"type": "bullish", "lower": float(lower), "upper": float(upper), "index": n}
        # Bearish gap: hi_next < lo_prev
        elif hi_next < lo_prev:
            lower = hi_next
            upper = lo_prev
            latest = {"type": "bearish", "lower": float(lower), "upper": float(upper), "index": n}
    return latest

def _price_retests_fvg(df: pd.DataFrame, side: str, atr_mult: float = RETEST_ATR_TOLERANCE) -> tuple[bool, str]:
    """
    Check whether price has retested the latest FVG consistent with the trade side.
    For BUY, prefer bullish FVG retest (close within gap +/- atr_mult*ATR).
    For SELL, prefer bearish FVG retest.
    Returns (ok, reason)
    """
    fvg = _detect_latest_fvg(df)
    if fvg is None:
        return False, "no fvg"

    last = df.iloc[-1]
    atr = float(last["atr"])
    close = float(last["close"])

    band_lower = fvg["lower"] - atr_mult * atr
    band_upper = fvg["upper"] + atr_mult * atr

    if side == "BUY" and fvg["type"] == "bullish":
        ok = band_lower <= close <= band_upper
        return ok, "bullish fvg retest" if ok else "not in bullish fvg band"
    if side == "SELL" and fvg["type"] == "bearish":
        ok = band_lower <= close <= band_upper
        return ok, "bearish fvg retest" if ok else "not in bearish fvg band"

    return False, "fvg type mismatch"

# ============================================================
# -------------------- SL/TP Calculation ---------------------
# ============================================================
def _compose_sl_from_sr_fvg(entry: float, atr: float, side: str, df: pd.DataFrame) -> float:
    """
    Build an SL that respects swing + FVG edge (if enabled).
    Always ensure MIN_SL_DIST_PCT away and add ATR buffers.
    """
    if side == "BUY":
        swing = _recent_support(df)
        sl_candidates = [swing - ATR_MULT_SL_BUFFER * atr]
        if USE_FVG_FOR_SL:
            fvg = _detect_latest_fvg(df)
            if fvg and fvg["type"] == "bullish":
                # SL under lower edge of bullish FVG
                sl_candidates.append(fvg["lower"] - ATR_MULT_SL_BUFFER * atr)
        sl = min(sl_candidates)  # further below price is safer for BUY
        # enforce minimum distance
        sl = min(entry - entry * MIN_SL_DIST_PCT, sl - EXTRA_SL_BUFFER_ATR * atr)
        return sl
    else:
        swing = _recent_resistance(df)
        sl_candidates = [swing + ATR_MULT_SL_BUFFER * atr]
        if USE_FVG_FOR_SL:
            fvg = _detect_latest_fvg(df)
            if fvg and fvg["type"] == "bearish":
                # SL above upper edge of bearish FVG
                sl_candidates.append(fvg["upper"] + ATR_MULT_SL_BUFFER * atr)
        sl = max(sl_candidates)  # further above price is safer for SELL
        sl = max(entry + entry * MIN_SL_DIST_PCT, sl + EXTRA_SL_BUFFER_ATR * atr)
        return sl

def calculate_stop_loss_and_tps(df: pd.DataFrame, side: str):
    last = df.iloc[-1]
    entry = float(last["close"])
    atr   = float(last["atr"])

    # NEW: stronger SL composition using SR+FVG
    sl = _compose_sl_from_sr_fvg(entry, atr, side, df)

    # Risk (R)
    risk = abs(entry - sl)
    if risk <= 0 or np.isnan(risk):
        return None, None

    # TPs as R-multiples
    tps = []
    for r in TP_R_MULTIPLES:
        if side == "BUY":
            tps.append(entry + r * risk)
        else:
            tps.append(entry - r * risk)

    # Guard against nonsense
    if any(np.isnan(x) for x in [sl, *tps]) or any(tp <= 0 for tp in tps):
        return None, None

    # For ordering
    tps = sorted(tps) if side == "BUY" else sorted(tps, reverse=True)
    return float(sl), [float(tp) for tp in tps]

# ============================================================
# -------------------- Strategy (Trigger + Filters) ----------
# ============================================================
def _macd_triggers(prev, last):
    cross_up   = (last["macd_line"] > last["macd_signal"]) and (prev["macd_line"] <= prev["macd_signal"])
    cross_down = (last["macd_line"] < last["macd_signal"]) and (prev["macd_line"] >= prev["macd_signal"])
    mom_up     = (last["macd_hist"] > 0) and (last["macd_hist"] > prev["macd_hist"])
    mom_down   = (last["macd_hist"] < 0) and (last["macd_hist"] < prev["macd_hist"])
    return cross_up or mom_up, cross_down or mom_down

def _quality_filters(df: pd.DataFrame, side: str) -> tuple[bool, list[str]]:
    """Return (is_ok, reasons)."""
    last = df.iloc[-1]
    reasons = []

    adx_ok = float(last["adx"]) >= MIN_ADX
    if not adx_ok:
        reasons.append("weak ADX")

    # Avoid chasing extended price vs EMA
    ema_short = float(last["ema_short"])
    atr = float(last["atr"])
    if side == "BUY":
        extended = (float(last["close"]) - ema_short) > EMA_EXTENSION_ATR * atr
    else:
        extended = (ema_short - float(last["close"])) > EMA_EXTENSION_ATR * atr
    if extended:
        reasons.append("extended from EMA")

    # Must be near SR in the correct direction
    if side == "BUY":
        level = _recent_support(df)
        dist_ok = _distance_to_level(float(last["close"]), level) <= MAX_DISTANCE_TO_LEVEL_ATR * atr
        if not dist_ok:
            reasons.append("far from support")
    else:
        level = _recent_resistance(df)
        dist_ok = _distance_to_level(float(last["close"]), level) <= MAX_DISTANCE_TO_LEVEL_ATR * atr
        if not dist_ok:
            reasons.append("far from resistance")

    return (adx_ok and not extended and dist_ok), (["ok"] if not reasons else reasons)

def _retest_ok_sr(df: pd.DataFrame, side: str, atr_mult: float = RETEST_ATR_TOLERANCE) -> bool:
    """
    SR retest check:
    - BUY: last close within atr_mult*ATR above recent support
    - SELL: last close within atr_mult*ATR below recent resistance
    """
    last = df.iloc[-1]
    atr = float(last["atr"])
    close = float(last["close"])
    if side == "BUY":
        support = _recent_support(df)
        return (support <= close <= support + atr_mult * atr)
    else:
        resistance = _recent_resistance(df)
        return (resistance - atr_mult * atr <= close <= resistance)

def _retest_ok(df: pd.DataFrame, side: str) -> tuple[bool, str]:
    """
    Combined retest check: SR OR FVG
    """
    sr_ok = _retest_ok_sr(df, side)
    if sr_ok:
        return True, "sr retest ok"
    fvg_ok, fvg_reason = _price_retests_fvg(df, side)
    if fvg_ok:
        return True, fvg_reason
    return False, "retest not validated"

def run_strategy(df: pd.DataFrame):
    """
    Returns: (decision: 'BUY'|'SELL'|'HOLD', confidence [0..100], reasons string)
    Confidence is additive but capped; main.py will gate by threshold.
    """
    if df is None or len(df) < 60:
        return "HOLD", 0.0, ""

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Primary triggers
    macd_buy, macd_sell = _macd_triggers(prev, last)

    buy_pts = 0.0
    sell_pts = 0.0
    buy_reasons = []
    sell_reasons = []

    # Give large weight to MACD trigger
    if macd_buy:
        buy_pts += 40
        buy_reasons.append("MACD trigger")
    if macd_sell:
        sell_pts += 40
        sell_reasons.append("MACD trigger")

    # EMA trend
    ema_bull = float(last["ema_short"]) > float(last["ema_long"])
    ema_bear = float(last["ema_short"]) < float(last["ema_long"])
    if ema_bull:
        buy_pts += 15
        buy_reasons.append("EMA trend up")
    if ema_bear:
        sell_pts += 15
        sell_reasons.append("EMA trend down")

    # RSI soft zone
    rsi = float(last["rsi"])
    if 38 <= rsi <= 70:
        buy_pts += 10
        buy_reasons.append("RSI ok for buy")
    if 30 <= rsi <= 62:
        sell_pts += 10
        sell_reasons.append("RSI ok for sell")

    # ADX/DI directional boost
    if float(last["adx"]) >= MIN_ADX:
        if float(last["adx_pos_di"]) > float(last["adx_neg_di"]):
            buy_pts += 15
            buy_reasons.append("ADX up")
        elif float(last["adx_neg_di"]) > float(last["adx_pos_di"]):
            sell_pts += 15
            sell_reasons.append("ADX down")

    # Volume confirmation (both sides)
    if float(last["volume"]) > float(last["volume_sma"]):
        buy_pts += 10
        sell_pts += 10

    # Quality filter (hard gate): must pass to consider a trade
    if macd_buy:
        ok, why = _quality_filters(df, "BUY")
        if ok:
            # Retest check happens *outside* here in pending logic, but we can add bonus if already met
            ret_ok, ret_reason = _retest_ok(df, "BUY")
            if ret_ok:
                buy_pts += 10
                buy_reasons.append("quality ok + retest")
            else:
                buy_reasons.append(ret_reason)
        else:
            buy_reasons += why
            buy_pts = 0  # block trade if quality fails
    if macd_sell:
        ok, why = _quality_filters(df, "SELL")
        if ok:
            ret_ok, ret_reason = _retest_ok(df, "SELL")
            if ret_ok:
                sell_pts += 10
                sell_reasons.append("quality ok + retest")
            else:
                sell_reasons.append(ret_reason)
        else:
            sell_reasons += why
            sell_pts = 0

    buy_conf = min(100.0, buy_pts)
    sell_conf = min(100.0, sell_pts)

    decision = "HOLD"
    chart_info = ""
    conf = 0.0

    if buy_conf > sell_conf and buy_conf > 0:
        decision = "BUY"
        conf = buy_conf
        chart_info = ", ".join(buy_reasons)
    elif sell_conf > buy_conf and sell_conf > 0:
        decision = "SELL"
        conf = sell_conf
        chart_info = ", ".join(sell_reasons)

    # Debug (printed by main logs)
    print(f"--- STRATEGY DEBUG for {df.name} ---")
    print(f" - Buy reasons: {buy_reasons} ({buy_conf:.2f}%)")
    print(f" - Sell reasons: {sell_reasons} ({sell_conf:.2f}%)")
    print(f" - Final Decision: {decision}, Confidence: {conf:.2f}%")
    print("------------------------------------")

    return decision, conf, chart_info

# ============================================================
# -------------------- Trade book-keeping --------------------
# ============================================================
class Trade:
    def __init__(self, symbol, direction, entry_price, sl, tps, confidence=None):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = float(entry_price)
        self.sl = float(sl)
        self.tps = [float(x) for x in (tps or [])]
        self.tp_hits = [False] * len(self.tps)
        self.active = True
        self.telegram_message_id = None
        self.entry_time = datetime.now()
        self.status = "OPEN"
        self.closed_time = None
        self.confidence = float(confidence) if confidence is not None else None
        self.candles_since_entry = 0
        # NEW
        self.use_trailing = ENABLE_ATR_TRAILING
        self.moved_to_be = False

    def __repr__(self):
        tps_str = ", ".join([f"{tp:.4f}" for tp in self.tps]) if self.tps else "[]"
        return (
            f"Trade(symbol={self.symbol}, direction={self.direction}, "
            f"entry={self.entry_price:.4f}, sl={self.sl:.4f}, tps=[{tps_str}], "
            f"confidence={self.confidence})"
        )

# ============================================================
# -------------- Reversal detection (for cancellation) -------
# ============================================================
def _opposite_signals(df: pd.DataFrame, trade: Trade) -> int:
    """
    Count how many indicators currently align AGAINST the trade direction.
    """
    if len(df) < 2:
        return 0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    macd_buy, macd_sell = _macd_triggers(prev, last)

    count = 0
    if trade.direction == "BUY":
        # Signals against BUY
        if macd_sell: count += 1
        if float(last["ema_short"]) < float(last["ema_long"]): count += 1
        if float(last["adx_neg_di"]) > float(last["adx_pos_di"]): count += 1
        if float(last["rsi"]) < 40: count += 1
        if float(last["close"]) < float(last["ema_short"]): count += 1
    else:
        # Signals against SELL
        if macd_buy: count += 1
        if float(last["ema_short"]) > float(last["ema_long"]): count += 1
        if float(last["adx_pos_di"]) > float(last["adx_neg_di"]): count += 1
        if float(last["rsi"]) > 60: count += 1
        if float(last["close"]) > float(last["ema_short"]): count += 1

    return count

# ============================================================
# -------------------- Telegram-facing helpers ---------------
# ============================================================
async def add_active_trade(
    symbol,
    entry_price,
    direction,
    stop_loss_price,
    take_profits_levels,
    active_trades_map,
    TradeClass,
    send_signal_alert_func,
    confidence,
    chart_pattern_info,
):
    if (
        entry_price is None
        or stop_loss_price is None
        or take_profits_levels is None
        or len(take_profits_levels) == 0
        or np.isnan(entry_price)
        or np.isnan(stop_loss_price)
    ):
        print(f"WARNING: Cannot create trade for {symbol}. Invalid entry/SL/TP.")
        return None

    trade = TradeClass(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        sl=stop_loss_price,
        tps=take_profits_levels,
        confidence=confidence,
    )
    active_trades_map[symbol] = trade

    try:
        msg_id = await send_signal_alert_func(trade, chart_pattern_info)
    except Exception as e:
        print(f"WARNING: Error while sending initial alert for {symbol}: {e}")
        msg_id = None

    if msg_id:
        trade.telegram_message_id = msg_id
        print(f"DEBUG: Stored Telegram message ID {msg_id} for trade {symbol}.")
    else:
        print(f"WARNING: Failed to send initial Telegram message for {symbol}.")

    return trade

async def check_and_update_active_trades(
    active_trades_map,
    df,
    send_sl_alert_func,
    send_tp_alert_func,
    send_final_alert_func
):
    symbol = df.name
    if symbol not in active_trades_map:
        return

    trade: Trade = active_trades_map[symbol]
    last = df.iloc[-1]
    atr = float(last["atr"])
    trade.candles_since_entry += 1

    # --- TP checks
    for i, tp in enumerate(trade.tps):
        if not trade.tp_hits[i]:
            if trade.direction == "BUY" and float(last["high"]) >= tp:
                trade.tp_hits[i] = True
                trade.status = "CLOSED_TP_PARTIAL"
                try:
                    await send_tp_alert_func(trade, i + 1, tp)
                except Exception as e:
                    print(f"WARNING: TP alert error for {symbol}: {e}")
            elif trade.direction == "SELL" and float(last["low"]) <= tp:
                trade.tp_hits[i] = True
                trade.status = "CLOSED_TP_PARTIAL"
                try:
                    await send_tp_alert_func(trade, i + 1, tp)
                except Exception as e:
                    print(f"WARNING: TP alert error for {symbol}: {e}")

    # --- NEW: Move SL to BE after TP index (default TP1)
    if MOVE_TO_BREAKEVEN_AFTER_TP_INDEX > 0 and any(trade.tp_hits):
        hit_count = sum(trade.tp_hits)
        if hit_count >= MOVE_TO_BREAKEVEN_AFTER_TP_INDEX and not trade.moved_to_be:
            old_sl = trade.sl
            if trade.direction == "BUY":
                trade.sl = max(trade.sl, trade.entry_price)  # to BE (or better)
            else:
                trade.sl = min(trade.sl, trade.entry_price)
            trade.moved_to_be = True
            print(f"[INFO] {symbol}: SL moved to BE (from {old_sl:.4f} to {trade.sl:.4f}) after TP{MOVE_TO_BREAKEVEN_AFTER_TP_INDEX}")

    # --- NEW: Optional ATR trailing (keeps room but protects)
    if ENABLE_ATR_TRAILING and any(trade.tp_hits):
        trail = ATR_TRAIL_MULT * atr
        old_sl = trade.sl
        if trade.direction == "BUY":
            candidate_sl = float(last["close"]) - trail
            trade.sl = max(trade.sl, candidate_sl)
        else:
            candidate_sl = float(last["close"]) + trail
            trade.sl = min(trade.sl, candidate_sl)
        if trade.sl != old_sl:
            print(f"[INFO] {symbol}: Trailing SL updated from {old_sl:.4f} to {trade.sl:.4f}")

    # --- SL check
    sl_hit = False
    if trade.direction == "BUY" and float(last["low"]) <= trade.sl:
        sl_hit = True
    if trade.direction == "SELL" and float(last["high"]) >= trade.sl:
        sl_hit = True

    if sl_hit:
        trade.active = False
        trade.closed_time = datetime.now()
        if any(trade.tp_hits):
            trade.status = "CLOSED_SL_AFTER_TP"
            final_msg = f"✅ Trade closed for {trade.symbol}.\nReason: Stop Loss after partial TP."
        else:
            trade.status = "CLOSED_SL"
            final_msg = f"❌ Trade closed for {trade.symbol}.\nReason: Stop Loss hit."
        try:
            await send_final_alert_func(trade, final_msg)
        except Exception as e:
            print(f"WARNING: Final SL alert error for {symbol}: {e}")
        del active_trades_map[symbol]
        return

    # --- Optional cancellation after entry if majority reverses
    if ENABLE_CANCEL_AFTER_ENTRY and trade.candles_since_entry >= CANCEL_AFTER_CANDLES:
        oppositions = _opposite_signals(df, trade)
        if oppositions >= REVERSAL_MAJORITY_THRESHOLD:
            trade.active = False
            trade.closed_time = datetime.now()
            trade.status = "CANCELLED_REVERSAL"
            msg = f"⚠️ Trade cancelled for {trade.symbol}.\nReason: Majority indicators reversed."
            try:
                await send_final_alert_func(trade, msg)
            except Exception as e:
                print(f"WARNING: Final cancel alert error for {symbol}: {e}")
            del active_trades_map[symbol]
            return

    # --- Close fully if all TPs hit
    if all(trade.tp_hits):
        trade.active = False
        trade.closed_time = datetime.now()
        trade.status = "CLOSED_TP_FULL"
        msg = f"🎉 Trade closed for {trade.symbol}.\nReason: All targets reached."
        try:
            await send_final_alert_func(trade, msg)
        except Exception as e:
            print(f"WARNING: Final TP alert error for {symbol}: {e}")
        del active_trades_map[symbol]

# ============================================================
# -------------------- Pending & Cooldown State --------------
# ============================================================
_last_signal_time = {}            # optional in-module cooldown (symbol -> datetime)
_pending_signals = {}             # symbol -> dict(pending info)
_daily_signal_count = {}          # symbol -> date_str -> count

def _today_key() -> str:
    # All times are naive; caller is typically running in one TZ. Good enough for throttling.
    return datetime.utcnow().strftime("%Y-%m-%d")

def _incr_daily_count(symbol: str):
    key = _today_key()
    entry = _daily_signal_count.get(symbol, {})
    entry[key] = entry.get(key, 0) + 1
    _daily_signal_count[symbol] = entry

def _daily_count(symbol: str) -> int:
    key = _today_key()
    entry = _daily_signal_count.get(symbol, {})
    return entry.get(key, 0)

def _record_pending(symbol: str, side: str, conf: float, reasons: str):
    _pending_signals[symbol] = {
        "side": side,
        "start_idx": None,     # we’ll fill with current candle index if needed
        "candles_waited": 0,
        "confidence": float(conf),
        "reasons": reasons,
        "created_at": datetime.utcnow(),
    }

def _clear_pending(symbol: str):
    if symbol in _pending_signals:
        del _pending_signals[symbol]

# ============================================================
# -------------------- Main prediction per symbol ------------
# ============================================================
async def predict_trade(
    exchange,
    symbol,
    active_trades_map,
    TradeClass,
    confidence_threshold,
    send_signal_alert_func,
    send_sl_alert_func,
    send_tp_alert_func,
    send_final_alert_func
):
    """
    Original API preserved. Adds:
    - pending MACD-first entries that wait for SR/FVG retest
    - throttling per day
    - cooldown
    """
    df = await fetch_ohlcv_data(exchange, symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
    if df is None or len(df) < 60:
        print(f"[INFO] Skipping {symbol}: insufficient data.")
        return

    df = calculate_indicators(df)
    if df is None:
        print(f"[INFO] Skipping {symbol}: indicators not ready.")
        return

    # Update active trades first
    await check_and_update_active_trades(
        active_trades_map, df, send_sl_alert_func, send_tp_alert_func, send_final_alert_func
    )

    # If there's already an active trade for this symbol, skip new signal
    if symbol in active_trades_map:
        print(f"[INFO] Active trade exists for {symbol}. Skipping new signal.")
        return

    # Optional local cooldown
    if SYMBOL_COOLDOWN_SECONDS > 0:
        last_t = _last_signal_time.get(symbol)
        if last_t and (datetime.now() - last_t).total_seconds() < SYMBOL_COOLDOWN_SECONDS:
            print(f"[INFO] Cooldown active for {symbol}. Skipping.")
            return

    # Signal throttling per day
    if MAX_SIGNALS_PER_DAY_PER_SYMBOL > 0 and _daily_count(symbol) >= MAX_SIGNALS_PER_DAY_PER_SYMBOL:
        print(f"[INFO] {symbol}: daily signal cap reached ({MAX_SIGNALS_PER_DAY_PER_SYMBOL}).")
        return

    # Run strategy (MACD-first + soft filters)
    decision, confidence, reasons = run_strategy(df)

    # If confidence too low, also try to release pending if they now retested
    if decision == "HOLD" or confidence < float(confidence_threshold):
        # Check if there is a pending trade waiting for retest that is now valid
        if REQUIRE_RETEST_AFTER_TRIGGER and symbol in _pending_signals:
            pend = _pending_signals[symbol]
            side = pend["side"]
            ret_ok, ret_reason = _retest_ok(df, side)
            # Optionally relax MACD on retest (user noted MACD often flips on retest)
            macd_buy, macd_sell = _macd_triggers(df.iloc[-2], df.iloc[-1])
            macd_supports = (macd_buy and side == "BUY") or (macd_sell and side == "SELL")
            if ret_ok and (macd_supports or ALLOW_WEAK_MACD_ON_RETEST):
                # Build SL/TP and emit signal now
                sl, tps = calculate_stop_loss_and_tps(df, side)
                if (sl is None) or (tps is None):
                    print(f"[WARNING] Failed SL/TP calc for {symbol} (pending release).")
                    _clear_pending(symbol)
                    return
                entry = float(df['close'].iloc[-1])
                if np.isnan(entry):
                    print(f"[WARNING] Entry NaN for {symbol} (pending release).")
                    _clear_pending(symbol)
                    return
                # Guard: SL not too tight
                min_sl_dist = entry * MIN_SL_DIST_PCT
                if abs(entry - sl) < min_sl_dist:
                    print(f"[WARNING] SL too close for {symbol} (pending release). Discarding.")
                    _clear_pending(symbol)
                    return

                # Send & store trade
                chart_reason = f"retest release: {ret_reason}"
                await add_active_trade(
                    symbol,
                    entry,
                    side,
                    sl,
                    tps,
                    active_trades_map,
                    TradeClass,
                    send_signal_alert_func,
                    pend["confidence"],
                    chart_reason,
                )
                _last_signal_time[symbol] = datetime.now()
                _clear_pending(symbol)
                _incr_daily_count(symbol)
                return

            # keep waiting but enforce timeout by candle count
            pend["candles_waited"] += 1
            if pend["candles_waited"] > PENDING_RETEST_MAX_CANDLES:
                print(f"[INFO] Pending expired for {symbol} after {PENDING_RETEST_MAX_CANDLES} candles.")
                _clear_pending(symbol)

        print(f"[INFO] No signal for {symbol}. Decision={decision}, Conf={confidence:.2f}%. Threshold={confidence_threshold}")
        return

    # --------------------------------------------------------
    # At this point, we have a BUY/SELL decision from scoring.
    # If we REQUIRE_RETEST_AFTER_TRIGGER and retest is NOT ready yet,
    # store a pending signal instead of entering right away.
    # --------------------------------------------------------
    if REQUIRE_RETEST_AFTER_TRIGGER:
        ret_ok, ret_reason = _retest_ok(df, decision)
        if not ret_ok:
            # store pending and return without sending a trade
            _record_pending(symbol, decision, confidence, reasons + f", waiting: {ret_reason}")
            print(f"[INFO] {symbol}: MACD trigger but waiting retest ({ret_reason}). Pending saved.")
            return
        else:
            reasons = f"{reasons}, {ret_reason}"

    # Compute SL/TP
    sl, tps = calculate_stop_loss_and_tps(df, decision)
    if sl is None or tps is None:
        print(f"[WARNING] Failed SL/TP calc for {symbol}. Skipping.")
        _clear_pending(symbol)
        return

    entry = float(df["close"].iloc[-1])
    if np.isnan(entry):
        print(f"[WARNING] Entry NaN for {symbol}.")
        _clear_pending(symbol)
        return

    # Guard: SL not too tight
    min_sl_dist = entry * MIN_SL_DIST_PCT
    if abs(entry - sl) < min_sl_dist:
        print(f"[WARNING] SL too close for {symbol}. Discarding.")
        _clear_pending(symbol)
        return

    # Send & store trade
    await add_active_trade(
        symbol,
        entry,
        decision,
        sl,
        tps,
        active_trades_map,
        TradeClass,
        send_signal_alert_func,
        confidence,
        reasons,
    )
    _last_signal_time[symbol] = datetime.now()
    _incr_daily_count(symbol)
    _clear_pending(symbol)

# ============================================================
# -------------------- END OF MODULE -------------------------
# ============================================================
