"""
trader.py – Autonomous Liquidity-Based Trading Engine
======================================================
Strategy:
  - Scans M5, M15, H1 for unswept BSL/SSL swing levels
  - Scores each level by multi-timeframe confluence + wick-rejection strength
  - Enters when the last closed M5 candle sweeps a level and closes back (reversal)
  - SL placed just beyond the sweep extreme + configurable tick buffer
  - TP placed at the nearest high-confluence opposing liquidity cluster
  - Trailing schedule:
      • 1.5 R  → SL to break-even
      • 2.5 R  → SL to 1:1
      • >2.5 R → trail SL 1 R behind price on every tick
  - Early exit: if a strong reversal candle closes against us before 1.5 R, close immediately
  - Capital: read live balance from broker, divide into N chunks, risk 1% per chunk per trade
"""

import asyncio
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd

from mt5_common import (
    apply_symbol_tuning,
    default_trader_symbol,
    ensure_mt5,
    is_xau_symbol,
    mt5_busy_hint,
    resolve_mt5_symbol,
    shutdown_mt5_if_owned,
    symbols_catalog,
)

# ═══════════════════════════════════════════════════════════
#  AGGRESSION  (1 = conservative  …  10 = aggressive)
#  Override with env: TRADER_AGGRESSION=7
#
#  What it scales:
#   min_confidence    – how strong the multi-TF confluence must be
#   min_rr            – minimum required reward-to-risk ratio
#   max_entry_drift   – how far price can be from the candle close on entry
#   loss_cooldown_s   – pause between trades after a close
#   allow_neutral_h1  – whether to trade in choppy / unclear H1 structure
# ═══════════════════════════════════════════════════════════
_AGG_RAW = int(os.environ.get("TRADER_AGGRESSION", "8"))
AGGRESSION: int = max(1, min(10, _AGG_RAW))   # clamp 1–10; default 8

# Lookup tables indexed by AGGRESSION (index 0 unused; 1–10 valid)
#                          1      2      3      4      5      6      7      8      9     10
_AGG_CONFIDENCE  = [0, 0.65,  0.60,  0.55,  0.48,  0.40,  0.35,  0.30,  0.25,  0.20, 0.15]
_AGG_MIN_RR      = [0,  3.5,   3.0,   2.5,   2.2,   2.0,   1.8,   1.6,   1.5,   1.3,  1.2]
_AGG_DRIFT_PCT   = [0, 4e-4,  5e-4,  6e-4,  7e-4,  8e-4,  1e-3, 1.3e-3,1.6e-3, 2e-3, 2.5e-3]
_AGG_COOLDOWN    = [0,  600,   540,   480,   390,   300,   240,   180,   120,    60,    30]
_AGG_NEUTRAL_H1  = [0, False, False, False, False, False, False,  True,  True,  True,  True]
# Trend scalp: fixed TP in R multiples (quick out)
_AGG_TREND_TP_R  = [0, 1.0,  1.1,  1.2,  1.3,  1.5,  1.6,  1.8,  2.0,  2.2,  2.5]
_AGG_TREND_MIN_RR = [0, 1.0,  1.0,  1.0,  1.1,  1.2,  1.2,  1.3,  1.4,  1.5,  1.5]

# ═══════════════════════════════════════════════════════════
#  CONFIG  – tune before running
# ═══════════════════════════════════════════════════════════
CFG: dict = {
    # ── Symbol (override with TRADER_SYMBOL env; default XAUUSD)
    "symbol":             default_trader_symbol(),

    # ── Timeframe scan ──────────────────────────────────────
    "scan_timeframes":    ["M5", "M15", "H1"],
    "entry_tf":           "M5",          # timeframe for sweep + reversal detection
    "candles": {
        "M5":  250,
        "M15": 200,
        "H1":  150,
    },
    "swing_window": {
        "M5":  5,
        "M15": 7,
        "H1":  10,
    },

    # ── Level clustering ────────────────────────────────────
    "cluster_pct":        0.0015,        # 0.15 % of mid price → nearby levels merged

    # ── Trade quality filters (scaled by AGGRESSION) ────────
    "min_confidence":     _AGG_CONFIDENCE[AGGRESSION],
    "min_rr":             _AGG_MIN_RR[AGGRESSION],
    "max_spread_pct":     0.003,

    # ── Entry timing / slippage gate (scaled by AGGRESSION) ─
    # How far current price can be from the sweep candle close.
    # Tight = precise entry, small SL.  Wide = take more setups, larger SL.
    "max_entry_drift_pct": _AGG_DRIFT_PCT[AGGRESSION],

    # ── Stop placement ──────────────────────────────────────
    "sl_buffer_ticks":    3,             # ticks beyond sweep extreme

    # ── H1 trend filter (scaled by AGGRESSION) ──────────────
    # When True, the engine also trades during choppy / NEUTRAL H1 structure.
    "allow_neutral_h1":   _AGG_NEUTRAL_H1[AGGRESSION],

    # ── WITHDRAWAL / BUFFER CAPITAL MODEL ───────────────────
    # This is NOT a compounding model.  Goal: extract profit to bank.
    #   • buffer_pct of session-start balance = daily budget
    #   • budget split into capital_chunks → risk per trade
    #   • Profit target  : session P&L ≥ +buffer → WITHDRAW ALERT
    #   • Daily loss limit: session P&L ≤ −buffer → STOP for the day
    "buffer_pct":         0.03,          # 3 % of broker balance = daily buffer
    "capital_chunks":     5,             # budget / 5 = risk per trade (~5 shots/day)
    "max_open_trades":    1,             # one position at a time

    # ── Trailing milestones (multiples of 1 R) ──────────────
    "be_at_r":            1.5,
    "trail_start_r":      2.5,

    # ── Early-exit reversal candle detection ────────────────
    "reversal_exit":      True,
    "reversal_body_pct":  0.50,          # body ≥ 50 % of range = strong candle

    # ── Cooldown after any close (scaled by AGGRESSION) ──────
    "loss_cooldown_s":    _AGG_COOLDOWN[AGGRESSION],

    # ── Trend momentum scalp (quick in / quick out with H1 direction) ──
    "trend_scalp_enabled": True,
    "trend_body_pct":      0.55,         # last M5 candle body ≥ 55 % of range
    "trend_pullback_pct":  0.0012,       # price within 0.12 % of EMA or support level
    "trend_tp_r":          _AGG_TREND_TP_R[AGGRESSION],
    "trend_min_rr":        _AGG_TREND_MIN_RR[AGGRESSION],
    "trend_be_at_r":       1.0,          # faster breakeven for scalps
    "trend_trail_start_r": 1.5,

    # ── Entry timing ────────────────────────────────────────
    # Scan all modes every tick; place market orders only right after M5 close (tight SL).
    "entry_window_after_close_s": 90,   # seconds after each M5 close to allow fills

    # ── Main loop cadence ───────────────────────────────────
    "loop_sleep":         2,             # seconds between ticks
}

MAGIC = 777001    # unique magic number; never clashes with server.py (123456)

# Print a live status line every this many seconds (even when idle)
HEARTBEAT_EVERY = 15

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

# Higher TF levels weigh more in the confluence score (max sum = 1+2+3 = 6)
TF_WEIGHT: Dict[str, int] = {"M5": 1, "M15": 2, "H1": 3}

# ═══════════════════════════════════════════════════════════
#  LOGGING
#  Console + file both show DEBUG so every engine decision is visible.
#  Third-party libs (MetaTrader5, asyncio, etc.) stay at WARNING to avoid noise.
# ═══════════════════════════════════════════════════════════
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

_file_h = logging.FileHandler("trader.log", encoding="utf-8")
_file_h.setFormatter(_fmt)
_file_h.setLevel(logging.DEBUG)

_con_h = logging.StreamHandler(sys.stdout)
_con_h.setFormatter(_fmt)
_con_h.setLevel(logging.DEBUG)

# Root stays at WARNING so MT5 internals don't flood the output
logging.basicConfig(level=logging.WARNING)

log = logging.getLogger("trader")
log.setLevel(logging.DEBUG)
log.addHandler(_file_h)
log.addHandler(_con_h)
log.propagate = False


# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════
@dataclass
class Level:
    price: float
    ltype: str                          # "BSL" (swing high) | "SSL" (swing low)
    timeframes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reaction: float = 0.0              # wick / body at pivot; higher = sharper rejection
    is_swept: bool = False             # already taken by price in the scanned window


@dataclass
class Setup:
    direction: str                     # "BUY" | "SELL"
    entry:     float
    sl:        float
    tp:        float
    volume:    float
    risk_usd:  float
    sweep_ext: float                   # actual candle low / high that swept the level
    confidence: float
    rr:        float
    entry_mode: str = "sweep"          # "sweep" | "trend"


@dataclass
class EntryContext:
    """Shared candle / trend / broker state for all entry modes."""
    symbol: str
    si: object
    tick: object
    trend: str
    candle_ts: int
    c_lo: float
    c_hi: float
    c_cl: float
    c_op: float
    point: float
    digits: int
    buf: float
    risk_usd: float
    max_drift: float


@dataclass
class ManagedPos:
    ticket:    int
    symbol:    str
    direction: str
    entry:     float
    sl:        float
    tp:        float
    volume:    float
    risk_usd:  float
    r_dist:    float                   # price distance of 1 R unit
    stage:     int = 0                 # 0=initial  1=BE moved  2=trailing
    entry_mode: str = "sweep"          # tighter trail schedule when "trend"


# ═══════════════════════════════════════════════════════════
#  MT5 HELPERS
# ═══════════════════════════════════════════════════════════
def normalize_vol(raw: float, si) -> float:
    step = float(si.volume_step) if si.volume_step else 0.01
    mn   = float(si.volume_min)  if si.volume_min  else step
    if step <= 0:
        step = 0.01
    vol = max(mn, math.floor(raw / step) * step)
    step_str = f"{step:.10f}".rstrip("0")
    dec = len(step_str.split(".")[1]) if "." in step_str else 2
    return round(vol, dec)


def get_candles(symbol: str, tf: str, count: int) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf], 0, count)
    if rates is None or len(rates) == 0:
        return None
    return pd.DataFrame(rates)


def calc_lot(symbol: str, direction: str, entry: float, sl: float, risk_usd: float, si) -> Optional[float]:
    """Broker-accurate lot sizing matching exact risk at SL."""
    ot = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    loss_one = mt5.order_calc_profit(ot, symbol, 1.0, entry, sl)
    if loss_one is None or abs(loss_one) < 1e-12:
        return None
    raw = risk_usd / abs(loss_one)
    vol = normalize_vol(raw, si)
    max_vol = float(si.volume_max or 1000)
    return min(vol, normalize_vol(max_vol, si))


def get_filling(si) -> int:
    FOK, IOC = 1, 2
    if si.filling_mode & FOK:
        return mt5.ORDER_FILLING_FOK
    if si.filling_mode & IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def session_buffer_usd(session_start_balance: float) -> float:
    """
    Daily buffer = 3 % of the broker balance at session open.
    This is the TOTAL amount allowed to be lost OR won today.
    """
    return round(session_start_balance * CFG["buffer_pct"], 2)


def chunk_risk_usd(session_start_balance: float) -> float:
    """Risk for one trade = buffer / chunks."""
    buf = session_buffer_usd(session_start_balance)
    return round(buf / CFG["capital_chunks"], 2)


# ═══════════════════════════════════════════════════════════
#  LIQUIDITY SCANNER
# ═══════════════════════════════════════════════════════════
def _wick_body_ratio(row: pd.Series, ltype: str) -> float:
    """How strong was the rejection at the pivot? wick/body > 1 = strong."""
    body = abs(float(row["close"]) - float(row["open"]))
    if ltype == "BSL":
        wick = float(row["high"]) - max(float(row["open"]), float(row["close"]))
    else:
        wick = min(float(row["open"]), float(row["close"])) - float(row["low"])
    return wick / max(body, 1e-9)


def scan_one_tf(df: pd.DataFrame, window: int, tf: str) -> List[Level]:
    levels: List[Level] = []
    seen: set = set()
    n = len(df)

    for i in range(window, n - window):
        hi = float(df.iloc[i]["high"])
        lo = float(df.iloc[i]["low"])

        # ── BSL: swing high – buy-side liquidity sitting above price ─────────
        if (hi > df.iloc[i - window:i]["high"].max() and
                hi >= df.iloc[i + 1:i + window + 1]["high"].max()):
            rp = round(hi, 5)
            if rp not in seen:
                seen.add(rp)
                swept = any(float(df.iloc[j]["high"]) > hi
                            for j in range(i + window + 1, n))
                rxn = _wick_body_ratio(df.iloc[i], "BSL")
                levels.append(Level(
                    price=hi, ltype="BSL", timeframes=[tf],
                    reaction=rxn, is_swept=swept,
                ))

        # ── SSL: swing low – sell-side liquidity sitting below price ─────────
        if (lo < df.iloc[i - window:i]["low"].min() and
                lo <= df.iloc[i + 1:i + window + 1]["low"].min()):
            rp = round(lo, 5)
            if rp not in seen:
                seen.add(rp)
                swept = any(float(df.iloc[j]["low"]) < lo
                            for j in range(i + window + 1, n))
                rxn = _wick_body_ratio(df.iloc[i], "SSL")
                levels.append(Level(
                    price=lo, ltype="SSL", timeframes=[tf],
                    reaction=rxn, is_swept=swept,
                ))

    return levels


def aggregate_levels(symbol: str) -> List[Level]:
    """
    Scan all configured timeframes, cluster nearby same-type levels,
    and score each cluster by TF confluence + reaction strength.
    Returns list sorted by confidence descending.
    """
    raw_levels: List[Level] = []
    for tf in CFG["scan_timeframes"]:
        df = get_candles(symbol, tf, CFG["candles"][tf])
        if df is None:
            log.warning("No candles for %s %s – skipping TF", symbol, tf)
            continue
        raw_levels.extend(scan_one_tf(df, CFG["swing_window"][tf], tf))

    if not raw_levels:
        return []

    tick = mt5.symbol_info_tick(symbol)
    mid  = ((tick.ask + tick.bid) / 2) if tick else 1.0
    radius = mid * CFG["cluster_pct"]

    used: List[bool] = [False] * len(raw_levels)
    clusters: List[Level] = []

    for i, base in enumerate(raw_levels):
        if used[i]:
            continue
        used[i] = True
        # Accumulate cluster: weighted price, best reaction, union of TFs
        w_price = base.price * TF_WEIGHT.get(base.timeframes[0], 1)
        w_total = TF_WEIGHT.get(base.timeframes[0], 1)
        tfs     = list(base.timeframes)
        rxn     = base.reaction
        swept   = base.is_swept

        for j in range(i + 1, len(raw_levels)):
            if used[j]:
                continue
            other = raw_levels[j]
            if other.ltype != base.ltype:
                continue
            if abs(other.price - base.price) > radius:
                continue
            used[j] = True
            w = TF_WEIGHT.get(other.timeframes[0], 1)
            w_price += other.price * w
            w_total += w
            tfs.extend(other.timeframes)
            rxn   = max(rxn, other.reaction)
            swept = swept and other.is_swept   # cluster is swept only if ALL swept

        cluster_price = w_price / w_total

        # Confidence formula:
        #   70 % from timeframe weight coverage (how many TFs see this level)
        #   30 % from wick-rejection strength at the pivot
        unique_tfs = list(dict.fromkeys(tfs))
        tf_score  = sum(TF_WEIGHT.get(t, 1) for t in unique_tfs)
        max_tf    = sum(TF_WEIGHT.values())                  # = 6 if all 3 TFs
        rxn_norm  = min(rxn / 2.0, 1.0)                     # clamp at 2× body
        confidence = 0.70 * (tf_score / max_tf) + 0.30 * rxn_norm

        clusters.append(Level(
            price=cluster_price,
            ltype=base.ltype,
            timeframes=unique_tfs,
            reaction=rxn,
            confidence=confidence,
            is_swept=swept,
        ))

    return sorted(clusters, key=lambda x: -x.confidence)


# ═══════════════════════════════════════════════════════════
#  H1 STRUCTURAL TREND FILTER
# ═══════════════════════════════════════════════════════════
def h1_trend(symbol: str) -> Optional[str]:
    """
    Read the last 60 H1 candles and determine structural trend using
    swing-high / swing-low sequence (market structure).

    Returns:
      "BULL"    – H1 is making higher highs and higher lows → only take BUYs
      "BEAR"    – H1 is making lower highs and lower lows  → only take SELLs
      "NEUTRAL" – mixed / choppy                           → no trades

    Method: find the last 3 confirmed swing highs and 3 swing lows on H1
    using a 3-bar window, then check whether the sequence is ascending,
    descending, or mixed.
    """
    df = get_candles(symbol, "H1", 80)
    if df is None or len(df) < 20:
        return "NEUTRAL"

    window = 3
    n = len(df)

    swing_highs: List[float] = []
    swing_lows:  List[float] = []

    for i in range(window, n - window):
        hi = float(df.iloc[i]["high"])
        lo = float(df.iloc[i]["low"])
        if (hi > df.iloc[i - window:i]["high"].max() and
                hi >= df.iloc[i + 1:i + window + 1]["high"].max()):
            swing_highs.append(hi)
        if (lo < df.iloc[i - window:i]["low"].min() and
                lo <= df.iloc[i + 1:i + window + 1]["low"].min()):
            swing_lows.append(lo)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL"

    # Use the last two of each to judge direction
    hh = swing_highs[-1] > swing_highs[-2]   # higher high
    hl = swing_lows[-1]  > swing_lows[-2]    # higher low
    lh = swing_highs[-1] < swing_highs[-2]   # lower high
    ll = swing_lows[-1]  < swing_lows[-2]    # lower low

    if hh and hl:
        return "BULL"
    if lh and ll:
        return "BEAR"
    # One condition met is a weak bias – treat as NEUTRAL to avoid bad trades
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════
#  ENTRY DETECTION
# ═══════════════════════════════════════════════════════════
def _find_tp(levels: List[Level], direction: str, entry: float, r_dist: float) -> Optional[Level]:
    """
    BUY  → nearest unswept BSL above (entry + min_rr × r_dist)
    SELL → nearest unswept SSL below (entry − min_rr × r_dist)
    Prefers closest level with confidence ≥ 0.30.
    """
    min_gap   = CFG["min_rr"] * r_dist
    tgt_type  = "BSL" if direction == "BUY" else "SSL"

    candidates = [
        lv for lv in levels
        if lv.ltype == tgt_type
        and not lv.is_swept
        and lv.confidence >= 0.30
        and (lv.price > entry + min_gap if direction == "BUY"
             else lv.price < entry - min_gap)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x.price - entry))


def last_closed_bar_ts(symbol: str, tf: Optional[str] = None) -> Optional[int]:
    """Unix open-time of the last fully closed bar on entry_tf (usually M5)."""
    tf = tf or CFG["entry_tf"]
    df = get_candles(symbol, tf, 3)
    if df is None or len(df) < 2:
        return None
    return int(df.iloc[-2]["time"])


def bar_tf_seconds(tf: Optional[str] = None) -> int:
    tf = tf or CFG["entry_tf"]
    return {"M1": 60, "M5": 300, "M15": 900, "H1": 3600}.get(tf, 300)


def seconds_until_bar_close(symbol: str, tf: Optional[str] = None) -> int:
    """Seconds until the currently forming bar closes."""
    tf = tf or CFG["entry_tf"]
    df = get_candles(symbol, tf, 3)
    if df is None or len(df) < 1:
        return bar_tf_seconds(tf)
    forming_ts = int(df.iloc[-1]["time"])
    return max(0, int(forming_ts + bar_tf_seconds(tf) - time.time()))


def _m5_ema(df: pd.DataFrame, period: int = 9) -> float:
    closes = df["close"].astype(float)
    return float(closes.ewm(span=period, adjust=False).mean().iloc[-1])


def _build_entry_context(
    symbol: str,
    session_start_balance: float,
    last_order_candle_ts: int,
) -> Optional[EntryContext]:
    tf = CFG["entry_tf"]
    df = get_candles(symbol, tf, 30)
    if df is None or len(df) < 12:
        log.debug("  [GATE] Not enough M5 candles")
        return None

    candle = df.iloc[-2]
    candle_ts = int(candle["time"])
    # Block only if we already placed an order on this closed bar (not on startup)
    if last_order_candle_ts > 0 and candle_ts == last_order_candle_ts:
        tf_sec = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600}.get(CFG["entry_tf"], 300)
        forming_ts = int(df.iloc[-1]["time"])
        wait_s = max(0, int(forming_ts + tf_sec - time.time()))
        log.info(
            "  [SKIP] Already placed on this %s bar – next close in ~%ds (not a confidence issue)",
            CFG["entry_tf"], wait_s,
        )
        return None

    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None:
        return None

    spread_pct = (tick.ask - tick.bid) / max(tick.bid, 1e-9)
    if spread_pct > CFG["max_spread_pct"]:
        log.info(
            "  [SKIP] Spread %.4f%% > max %.4f%% – waiting for tighter spread",
            spread_pct * 100, CFG["max_spread_pct"] * 100,
        )
        return None

    trend = h1_trend(symbol)
    if trend == "NEUTRAL" and not CFG["allow_neutral_h1"]:
        log.info(
            "  [SKIP] H1 = NEUTRAL – raise AGGRESSION ≥ 7 for neutral, "
            "or wait for clear BULL/BEAR structure",
        )
        return None

    c_cl = float(candle["close"])
    log.info(
        "  [GATE] H1=%s │ M5 close=%.2f │ open=%.2f │ high=%.2f │ low=%.2f │ ts=%d",
        trend, c_cl, float(candle["open"]), float(candle["high"]), float(candle["low"]), candle_ts,
    )

    return EntryContext(
        symbol=symbol,
        si=si,
        tick=tick,
        trend=trend,
        candle_ts=candle_ts,
        c_lo=float(candle["low"]),
        c_hi=float(candle["high"]),
        c_cl=c_cl,
        c_op=float(candle["open"]),
        point=float(si.point),
        digits=int(si.digits),
        buf=CFG["sl_buffer_ticks"] * float(si.point),
        risk_usd=chunk_risk_usd(session_start_balance),
        max_drift=c_cl * CFG["max_entry_drift_pct"],
    )


def _detect_sweep_reversal(ctx: EntryContext, levels: List[Level]) -> Optional[Setup]:
    """Mode 1: sweep liquidity level + close back (reversal at SSL/BSL)."""
    symbol = ctx.symbol
    si, tick = ctx.si, ctx.tick
    trend = ctx.trend
    c_lo, c_hi, c_cl = ctx.c_lo, ctx.c_hi, ctx.c_cl
    buf, digits = ctx.buf, ctx.digits
    risk_usd, max_drift = ctx.risk_usd, ctx.max_drift

    # ── BUY: swept SSL, closed back above ───────────────────────────────
    if trend in ("BULL", "NEUTRAL"):
        for lv in levels:
            if lv.ltype != "SSL" or lv.is_swept:
                continue
            if lv.confidence < CFG["min_confidence"]:
                continue
            if not (c_lo < lv.price and c_cl > lv.price):
                continue

            entry = tick.ask

            # Gate: entry must be within max_drift of the candle close.
            drift = entry - c_cl
            if drift > max_drift:
                log.info(
                    "  [SKIP] BUY @ SSL %.2f – price drifted %.1f pts above close %.2f "
                    "(max allowed %.1f pts) – too late to enter",
                    lv.price, drift, c_cl, max_drift,
                )
                continue
            if drift < 0:
                log.info(
                    "  [SKIP] BUY @ SSL %.2f – price %.2f pulled back below candle close %.2f",
                    lv.price, entry, c_cl,
                )
                continue

            sl     = round(c_lo - buf, digits)
            r_dist = entry - sl
            if r_dist <= 0:
                continue

            tp_lv = _find_tp(levels, "BUY", entry, r_dist)
            if tp_lv is None:
                log.info(
                    "  [SKIP] BUY @ SSL %.2f – no unswept BSL target ≥ %.1f R above entry",
                    lv.price, CFG["min_rr"],
                )
                continue

            rr = (tp_lv.price - entry) / r_dist
            if rr < CFG["min_rr"]:
                log.info(
                    "  [SKIP] BUY @ SSL %.2f – R:R %.2f < min %.2f "
                    "(entry=%.2f sl=%.2f tp=%.2f)",
                    lv.price, rr, CFG["min_rr"], entry, sl, tp_lv.price,
                )
                continue

            vol = calc_lot(symbol, "BUY", entry, sl, risk_usd, si)
            if vol is None:
                log.warning("Could not compute lot for BUY setup")
                continue

            log.info(
                "  [SWEEP] BUY │ swept SSL %.5f │ close %.5f │ entry %.5f │ "
                "sl %.5f │ tp %.5f │ drift %.1f │ R:R %.1f │ conf %.2f",
                lv.price, c_cl, entry, sl, tp_lv.price, drift, rr, lv.confidence,
            )
            return Setup(
                "BUY", entry, sl, tp_lv.price, vol, risk_usd,
                sweep_ext=c_lo, confidence=lv.confidence, rr=rr, entry_mode="sweep",
            )

    # ── SELL: swept BSL, closed back below ──────────────────────────────
    # Allowed when H1=BEAR, or H1=NEUTRAL with aggression ≥ 7
    if trend in ("BEAR", "NEUTRAL"):
        for lv in levels:
            if lv.ltype != "BSL" or lv.is_swept:
                continue
            if lv.confidence < CFG["min_confidence"]:
                continue
            if not (c_hi > lv.price and c_cl < lv.price):
                continue

            entry = tick.bid

            drift = c_cl - entry
            if drift > max_drift:
                log.info(
                    "  [SKIP] SELL @ BSL %.2f – price drifted %.1f pts below close %.2f "
                    "(max %.1f pts) – too late to enter",
                    lv.price, drift, c_cl, max_drift,
                )
                continue
            if drift < 0:
                log.info(
                    "  [SKIP] SELL @ BSL %.2f – price %.2f pulled back above candle close %.2f",
                    lv.price, entry, c_cl,
                )
                continue

            sl     = round(c_hi + buf, digits)
            r_dist = sl - entry
            if r_dist <= 0:
                continue

            tp_lv = _find_tp(levels, "SELL", entry, r_dist)
            if tp_lv is None:
                log.info(
                    "  [SKIP] SELL @ BSL %.2f – no unswept SSL target ≥ %.1f R below entry",
                    lv.price, CFG["min_rr"],
                )
                continue

            rr = (entry - tp_lv.price) / r_dist
            if rr < CFG["min_rr"]:
                log.info(
                    "  [SKIP] SELL @ BSL %.2f – R:R %.2f < min %.2f "
                    "(entry=%.2f sl=%.2f tp=%.2f)",
                    lv.price, rr, CFG["min_rr"], entry, sl, tp_lv.price,
                )
                continue

            vol = calc_lot(symbol, "SELL", entry, sl, risk_usd, si)
            if vol is None:
                log.warning("Could not compute lot for SELL setup")
                continue

            log.info(
                "  [SWEEP] SELL │ swept BSL %.5f │ close %.5f │ entry %.5f │ "
                "sl %.5f │ tp %.5f │ drift %.1f │ R:R %.1f │ conf %.2f",
                lv.price, c_cl, entry, sl, tp_lv.price, drift, rr, lv.confidence,
            )
            return Setup(
                "SELL", entry, sl, tp_lv.price, vol, risk_usd,
                sweep_ext=c_hi, confidence=lv.confidence, rr=rr, entry_mode="sweep",
            )

    log.debug("  [SWEEP] No liquidity sweep+reversal on this candle (H1=%s)", trend)
    return None


def _try_trend_buy_scalp(
    ctx: EntryContext,
    levels: List[Level],
    *,
    h1: str,
    ema: float,
    swing_lo: float,
    body_pct: float,
    align: str,
) -> Optional[Setup]:
    """BUY: momentum follows bullish M5; enter at SSL or EMA support."""
    symbol, si, tick = ctx.symbol, ctx.si, ctx.tick
    mid = (tick.ask + tick.bid) / 2
    pullback_band = ctx.c_cl * CFG["trend_pullback_pct"]
    risk_usd, buf, digits = ctx.risk_usd, ctx.buf, ctx.digits

    supports = [
        lv for lv in levels
        if lv.ltype == "SSL" and not lv.is_swept
        and lv.confidence >= CFG["min_confidence"] * 0.8
    ]
    at_ema = abs(mid - ema) <= pullback_band
    at_ssl = any(abs(mid - lv.price) <= pullback_band for lv in supports)
    if not at_ema and not at_ssl:
        log.info(
            "  [TREND] Skip BUY (%s) – not at SSL/EMA (mid=%.2f ema=%.2f)",
            align, mid, ema,
        )
        return None

    entry = tick.ask
    sl = round(min(ctx.c_lo, swing_lo) - buf, digits)
    r_dist = entry - sl
    if r_dist <= 0:
        return None

    tp_scalp = entry + CFG["trend_tp_r"] * r_dist
    tp_lv = _find_tp(levels, "BUY", entry, r_dist)
    tp = min(tp_scalp, tp_lv.price) if tp_lv else tp_scalp
    rr = (tp - entry) / r_dist
    if rr < CFG["trend_min_rr"]:
        log.info("  [TREND] Skip BUY (%s) – R:R %.2f < min %.2f", align, rr, CFG["trend_min_rr"])
        return None

    vol = calc_lot(symbol, "BUY", entry, sl, risk_usd, si)
    if vol is None:
        return None

    anchor = "EMA" if at_ema else "SSL"
    log.info(
        "  [TREND] BUY scalp │ H1=%s │ %s │ M5 bullish │ @%s │ body=%.0f%% │ "
        "entry %.5f sl %.5f tp %.5f │ R:R %.2f",
        h1, align, anchor, body_pct * 100, entry, sl, tp, rr,
    )
    return Setup(
        "BUY", entry, sl, tp, vol, risk_usd,
        sweep_ext=ctx.c_lo, confidence=0.75, rr=rr, entry_mode="trend",
    )


def _try_trend_sell_scalp(
    ctx: EntryContext,
    levels: List[Level],
    *,
    h1: str,
    ema: float,
    swing_hi: float,
    body_pct: float,
    align: str,
) -> Optional[Setup]:
    """SELL: momentum follows bearish M5; enter at BSL or EMA resistance."""
    symbol, si, tick = ctx.symbol, ctx.si, ctx.tick
    mid = (tick.ask + tick.bid) / 2
    pullback_band = ctx.c_cl * CFG["trend_pullback_pct"]
    risk_usd, buf, digits = ctx.risk_usd, ctx.buf, ctx.digits

    resists = [
        lv for lv in levels
        if lv.ltype == "BSL" and not lv.is_swept
        and lv.confidence >= CFG["min_confidence"] * 0.8
    ]
    at_ema = abs(mid - ema) <= pullback_band
    at_bsl = any(abs(mid - lv.price) <= pullback_band for lv in resists)
    if not at_ema and not at_bsl:
        log.info(
            "  [TREND] Skip SELL (%s) – not at BSL/EMA (mid=%.2f ema=%.2f)",
            align, mid, ema,
        )
        return None

    entry = tick.bid
    sl = round(max(ctx.c_hi, swing_hi) + buf, digits)
    r_dist = sl - entry
    if r_dist <= 0:
        return None

    tp_scalp = entry - CFG["trend_tp_r"] * r_dist
    tp_lv = _find_tp(levels, "SELL", entry, r_dist)
    tp = max(tp_scalp, tp_lv.price) if tp_lv else tp_scalp
    rr = (entry - tp) / r_dist
    if rr < CFG["trend_min_rr"]:
        log.info("  [TREND] Skip SELL (%s) – R:R %.2f < min %.2f", align, rr, CFG["trend_min_rr"])
        return None

    vol = calc_lot(symbol, "SELL", entry, sl, risk_usd, si)
    if vol is None:
        return None

    anchor = "EMA" if at_ema else "BSL"
    log.info(
        "  [TREND] SELL scalp │ H1=%s │ %s │ M5 bearish │ @%s │ body=%.0f%% │ "
        "entry %.5f sl %.5f tp %.5f │ R:R %.2f",
        h1, align, anchor, body_pct * 100, entry, sl, tp, rr,
    )
    return Setup(
        "SELL", entry, sl, tp, vol, risk_usd,
        sweep_ext=ctx.c_hi, confidence=0.75, rr=rr, entry_mode="trend",
    )


def _detect_trend_scalp(ctx: EntryContext, levels: List[Level]) -> Optional[Setup]:
    """
    Mode 2: follow the M5 candle direction (quick in / out).

    Bullish M5 → look for BUY (at SSL/EMA)
    Bearish M5 → look for SELL (at BSL/EMA)

    H1 trend labels the setup as with-trend or counter-trend but does NOT block
    the opposite side — e.g. H1=BULL + bearish M5 still evaluates a SELL scalp.
    """
    if not CFG["trend_scalp_enabled"]:
        log.debug("  [TREND] Disabled in config")
        return None

    h1 = ctx.trend
    symbol = ctx.symbol
    tick = ctx.tick
    df = get_candles(symbol, CFG["entry_tf"], 30)
    if df is None or len(df) < 12:
        return None

    ema = _m5_ema(df, 9)
    body = abs(ctx.c_cl - ctx.c_op)
    rng = ctx.c_hi - ctx.c_lo
    if rng <= 0:
        return None
    body_pct = body / rng
    if body_pct < CFG["trend_body_pct"]:
        log.info(
            "  [TREND] Skip – last M5 body %.0f%% < %.0f%% (weak momentum)",
            body_pct * 100, CFG["trend_body_pct"] * 100,
        )
        return None

    recent = df.iloc[-10:-2]
    swing_lo = float(recent["low"].min())
    swing_hi = float(recent["high"].max())

    m5_bullish = ctx.c_cl > ctx.c_op
    m5_bearish = ctx.c_cl < ctx.c_op
    if not m5_bullish and not m5_bearish:
        log.info("  [TREND] Skip – doji M5 candle (no direction)")
        return None

    if m5_bullish:
        align = "with H1" if h1 == "BULL" else ("counter H1" if h1 == "BEAR" else "neutral H1")
        log.info("  [TREND] M5 bullish → checking BUY scalp (%s) ...", align)
        setup = _try_trend_buy_scalp(
            ctx, levels, h1=h1, ema=ema, swing_lo=swing_lo,
            body_pct=body_pct, align=align,
        )
        if setup:
            return setup

    if m5_bearish:
        align = "with H1" if h1 == "BEAR" else ("counter H1" if h1 == "BULL" else "neutral H1")
        log.info("  [TREND] M5 bearish → checking SELL scalp (%s) ...", align)
        setup = _try_trend_sell_scalp(
            ctx, levels, h1=h1, ema=ema, swing_hi=swing_hi,
            body_pct=body_pct, align=align,
        )
        if setup:
            return setup

    return None


def detect_setup(
    symbol: str,
    levels: List[Level],
    session_start_balance: float,
    last_order_candle_ts: int,
    *,
    quiet: bool = False,
) -> Optional[Setup]:
    """
    Evaluate the bar that just closed (caller must gate on M5 bar change).

    Signals (first match wins):
      1. Sweep + reversal — liquidity level taken then price closes back through it
      2. Trend scalp — bullish M5 close → BUY at SSL/EMA
      3. Trend scalp — bearish M5 close → SELL at BSL/EMA
    """
    ctx = _build_entry_context(symbol, session_start_balance, last_order_candle_ts)
    if ctx is None:
        return None

    m5_dir = "bullish" if ctx.c_cl > ctx.c_op else ("bearish" if ctx.c_cl < ctx.c_op else "doji")
    _log = log.debug if quiet else log.info
    _log(
        "  [ENTRY] Closed bar │ %s │ O=%.2f H=%.2f L=%.2f C=%.2f │ H1=%s",
        m5_dir, ctx.c_op, ctx.c_hi, ctx.c_lo, ctx.c_cl, ctx.trend,
    )

    _log("  [ENTRY] (1/3) Liquidity swept + reversed ...")
    setup = _detect_sweep_reversal(ctx, levels)
    if setup:
        return setup

    _log("  [ENTRY] (2/3) M5 direction scalp (bullish→BUY / bearish→SELL) ...")
    setup = _detect_trend_scalp(ctx, levels)
    if setup:
        return setup

    _log(
        "  [ENTRY] No trade this bar – no sweep/reversal and no valid %s scalp",
        m5_dir,
    )
    return None


# ═══════════════════════════════════════════════════════════
#  ORDER EXECUTION HELPERS
# ═══════════════════════════════════════════════════════════
def place_order(symbol: str, setup: Setup) -> Optional[int]:
    si   = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None:
        return None

    price = tick.ask if setup.direction == "BUY" else tick.bid
    ot    = mt5.ORDER_TYPE_BUY if setup.direction == "BUY" else mt5.ORDER_TYPE_SELL

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       setup.volume,
        "type":         ot,
        "price":        price,
        "sl":           round(setup.sl, si.digits),
        "tp":           round(setup.tp, si.digits),
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      f"{setup.entry_mode}|rr={setup.rr:.1f}|c={setup.confidence:.2f}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": get_filling(si),
    }
    res = mt5.order_send(req)
    if res is None:
        log.error("order_send returned None (terminal timeout?)")
        return None
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Order rejected: retcode=%d  comment=%s", res.retcode, res.comment)
        return None
    return int(res.order)


def _modify_sl(pos: ManagedPos, new_sl: float) -> bool:
    si = mt5.symbol_info(pos.symbol)
    if si is None:
        return False
    new_sl = round(new_sl, si.digits)

    # Guard: never move SL against us
    if pos.direction == "BUY"  and new_sl < pos.sl - 1e-9:
        return False
    if pos.direction == "SELL" and new_sl > pos.sl + 1e-9:
        return False

    res = mt5.order_send({
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "symbol":   pos.symbol,
        "sl":       new_sl,
        "tp":       round(pos.tp, si.digits),
    })
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        pos.sl = new_sl
        return True
    log.warning("Modify SL failed for ticket %d: %s", pos.ticket,
                res.comment if res else "None")
    return False


def _close_position(pos: ManagedPos, reason: str) -> bool:
    si   = mt5.symbol_info(pos.symbol)
    tick = mt5.symbol_info_tick(pos.symbol)
    if si is None or tick is None:
        return False

    close_type = mt5.ORDER_TYPE_SELL if pos.direction == "BUY" else mt5.ORDER_TYPE_BUY
    price      = tick.bid if pos.direction == "BUY" else tick.ask

    res = mt5.order_send({
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     pos.ticket,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "price":        price,
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      reason[:31],
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": get_filling(si),
    })
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        log.info("Closed ticket %d (%s): %s", pos.ticket, pos.direction, reason)
    else:
        log.warning("Close failed for %d: %s", pos.ticket,
                    res.comment if res else "None")
    return ok


# ═══════════════════════════════════════════════════════════
#  POSITION MANAGER
# ═══════════════════════════════════════════════════════════
class PositionManager:
    """
    Tracks all engine-managed positions and applies the trailing schedule:

    Stage 0  (open → 1.5 R)
        • watch for reversal candle → early exit if triggered
    Stage 1  (1.5 R hit)
        • SL moved to entry (breakeven)
    Stage 2  (2.5 R hit)
        • SL moved to entry + 1 R (1:1 locked)
        • from here, trail SL 1 R below current price on every update
    """

    def __init__(self):
        self.positions: Dict[int, ManagedPos] = {}

    def add(self, ticket: int, setup: Setup, symbol: str):
        r_dist = abs(setup.entry - setup.sl)
        pos = ManagedPos(
            ticket    = ticket,
            symbol    = symbol,
            direction = setup.direction,
            entry     = setup.entry,
            sl        = setup.sl,
            tp        = setup.tp,
            volume    = setup.volume,
            risk_usd  = setup.risk_usd,
            r_dist     = r_dist,
            entry_mode = setup.entry_mode,
        )
        self.positions[ticket] = pos
        log.info(
            "Tracking %-4s [%s] ticket=%-10d  entry=%.5f  sl=%.5f  tp=%.5f  "
            "vol=%.3f  rr=%.1f  conf=%.2f  risk=$%.2f",
            setup.direction, setup.entry_mode, ticket, setup.entry, setup.sl, setup.tp,
            setup.volume, setup.rr, setup.confidence, setup.risk_usd,
        )

    def sync(self, symbol: str):
        """Remove positions closed by broker SL/TP/manual intervention."""
        live = {
            int(p.ticket)
            for p in (mt5.positions_get(symbol=symbol) or ())
            if int(p.magic) == MAGIC
        }
        gone = [t for t in self.positions if t not in live]
        for t in gone:
            log.info("Ticket %d no longer open (closed by broker/SL/TP)", t)
            del self.positions[t]
        return gone

    def update_all(self, symbol: str) -> List[int]:
        """
        Apply trailing / early-exit logic for every tracked position.
        Returns list of tickets that were force-closed by early exit.
        """
        early_closed: List[int] = []
        for ticket, pos in list(self.positions.items()):
            pos.symbol = symbol
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            price = tick.ask if pos.direction == "BUY" else tick.bid
            ratio = self._gained_r(pos, price)
            # Live floating P&L from broker
            broker_pos = next(
                (p for p in (mt5.positions_get(symbol=symbol) or [])
                 if int(p.ticket) == ticket), None
            )
            floating = float(broker_pos.profit) if broker_pos else 0.0
            stage_label = ["unprotected", "breakeven", "trailing"][min(pos.stage, 2)]
            log.debug(
                "  [POS] ticket=%d │ %s │ %.2f R │ float=$%.2f │ sl=%.2f │ stage=%s",
                ticket, pos.direction, ratio, floating, pos.sl, stage_label,
            )
            self._manage(pos, price, early_closed)
        return early_closed

    # ── internal ──────────────────────────────────────────────────────────
    def _gained_r(self, pos: ManagedPos, price: float) -> float:
        gained = (price - pos.entry) if pos.direction == "BUY" else (pos.entry - price)
        return gained / pos.r_dist if pos.r_dist > 0 else 0.0

    def _manage(self, pos: ManagedPos, price: float, closed_out: List[int]):
        ratio = self._gained_r(pos, price)
        r     = pos.r_dist
        if pos.entry_mode == "trend":
            be_at = CFG["trend_be_at_r"]
            trail_start = CFG["trend_trail_start_r"]
        else:
            be_at = CFG["be_at_r"]
            trail_start = CFG["trail_start_r"]

        if pos.stage == 0:
            # Early-exit check while still unprotected
            if CFG["reversal_exit"] and self._reversal_detected(pos):
                if _close_position(pos, "early-exit-reversal"):
                    closed_out.append(pos.ticket)
                    del self.positions[pos.ticket]
                return

            if ratio >= be_at:
                if _modify_sl(pos, pos.entry):
                    pos.stage = 1
                    log.info(
                        "Ticket %d [%s] │ %.1f R │ SL → breakeven %.5f",
                        pos.ticket, pos.entry_mode, ratio, pos.entry,
                    )

        elif pos.stage == 1:
            if ratio >= trail_start:
                lock_sl = (pos.entry + r) if pos.direction == "BUY" else (pos.entry - r)
                if _modify_sl(pos, lock_sl):
                    pos.stage = 2
                    log.info(
                        "Ticket %d │ %.1f R │ SL locked at 1:1 %.5f",
                        pos.ticket, ratio, lock_sl,
                    )

        elif pos.stage == 2:
            # Trail: keep SL exactly 1 R behind current price
            trail = (price - r) if pos.direction == "BUY" else (price + r)
            if pos.direction == "BUY"  and trail > pos.sl + r * 0.10:
                if _modify_sl(pos, trail):
                    log.info(
                        "  [TRAIL] ticket=%d │ %.2f R │ SL trailed → %.5f",
                        pos.ticket, ratio, trail,
                    )
            elif pos.direction == "SELL" and trail < pos.sl - r * 0.10:
                if _modify_sl(pos, trail):
                    log.info(
                        "  [TRAIL] ticket=%d │ %.2f R │ SL trailed → %.5f",
                        pos.ticket, ratio, trail,
                    )

    def _reversal_detected(self, pos: ManagedPos) -> bool:
        """
        Strong reversal candle while in stage 0 (no breakeven yet):
        BUY  → bearish body ≥ 50 % of candle range AND close < entry
        SELL → bullish body ≥ 50 % of candle range AND close > entry
        """
        df = get_candles(pos.symbol, CFG["entry_tf"], 4)
        if df is None or len(df) < 2:
            return False
        c    = df.iloc[-2]   # last fully closed candle
        lo   = float(c["low"])
        hi   = float(c["high"])
        op   = float(c["open"])
        cl   = float(c["close"])
        body = abs(cl - op)
        rng  = hi - lo

        threshold = CFG["reversal_body_pct"]

        if pos.direction == "BUY":
            strong_bear = cl < op and rng > 0 and body / rng >= threshold
            return strong_bear and cl < pos.entry
        else:
            strong_bull = cl > op and rng > 0 and body / rng >= threshold
            return strong_bull and cl > pos.entry


# ═══════════════════════════════════════════════════════════
#  TRADING ENGINE  (orchestrator)
# ═══════════════════════════════════════════════════════════
class TradingEngine:
    """
    Withdrawal-based capital model:

      session_start_balance – broker balance when the engine first starts today
      buffer_usd            – 3 % of that balance  (≈ $40 on $1233 net worth)
      chunk_usd             – buffer / 10           (risk per individual trade)

    State machine:
      RUNNING   → normal operation; entries allowed
      WITHDRAW  → profit target hit; log loud alert; stop new entries; wait for human
      STOPPED   → daily loss limit hit; no new entries until next session
    """

    def __init__(self, reuse_mt5: bool = False):
        self.symbol: str = ""
        self.pm = PositionManager()
        self._last_close_ts: float   = 0.0
        self._known_tickets: set     = set()
        self._reuse_mt5              = reuse_mt5
        self._we_initialized_mt5     = False

        # Buffer / withdrawal tracking
        self._session_start_balance: float = 0.0
        self._buffer_usd:            float = 0.0
        self._session_pnl:           float = 0.0   # running P&L vs start balance
        self._state: str                   = "RUNNING"   # RUNNING | WITHDRAW | STOPPED

        # Only set after a real order fill — blocks duplicate orders same M5 bar
        self._last_order_candle_ts: int = 0
        self._last_closed_candle_ts: int = 0
        self._entry_window_until: float = 0.0   # unix time – market orders allowed until then

    # ── startup ───────────────────────────────────────────────────────────
    def _init(self) -> bool:
        symbol_hint = os.environ.get("TRADER_SYMBOL", CFG["symbol"]).strip()

        ok, we_init = ensure_mt5()
        self._we_initialized_mt5 = we_init and not self._reuse_mt5
        if not ok:
            log.error("MT5 init failed: %s", mt5.last_error())
            return False

        if not symbols_catalog():
            log.error("MT5 connected but symbol list is empty. %s", mt5_busy_hint())
            return False

        resolved = resolve_mt5_symbol(symbol_hint)
        if resolved is None:
            needle = "XAU" if is_xau_symbol(symbol_hint) else "BTC"
            sample = [
                s.name for s in symbols_catalog()
                if needle in s.name.upper() or "GOLD" in s.name.upper()
            ][:8]
            log.error(
                "Symbol %r not found in MT5. %s Matching symbols: %s",
                symbol_hint, mt5_busy_hint(), sample or "(none)",
            )
            return False

        self.symbol = resolved
        mt5.symbol_select(self.symbol, True)
        apply_symbol_tuning(CFG, self.symbol, _AGG_DRIFT_PCT[AGGRESSION])
        CFG["symbol"] = self.symbol
        log.info(
            "Instrument profile: %s │ cluster=%.2f%% │ max_spread=%.3f%% │ drift=%.3f%% │ sl_buf=%d ticks",
            "XAU" if is_xau_symbol(self.symbol) else "BTC/OTHER",
            CFG["cluster_pct"] * 100,
            CFG["max_spread_pct"] * 100,
            CFG["max_entry_drift_pct"] * 100,
            CFG["sl_buffer_ticks"],
        )

        seed_ts = last_closed_bar_ts(self.symbol)
        if seed_ts is not None:
            self._last_closed_candle_ts = seed_ts
            # If we restarted shortly after a close, stay in that bar's entry window
            bar_age = time.time() - seed_ts
            win = CFG["entry_window_after_close_s"]
            if bar_age < win:
                self._entry_window_until = time.time() + (win - bar_age)
                log.info(
                    "  [M5] In entry window for %.0fs more on current bar (then wait ~%ds for next close)",
                    self._entry_window_until - time.time(),
                    max(0, seconds_until_bar_close(self.symbol)),
                )
            else:
                log.info(
                    "  [M5] Scanning every tick │ orders only for %ds after each %s close │ next close ~%ds",
                    win, CFG["entry_tf"], seconds_until_bar_close(self.symbol),
                )

        ai = mt5.account_info()
        if ai is None:
            log.error("account_info() failed – cannot read balance")
            return False

        self._session_start_balance = float(ai.balance)
        self._buffer_usd = session_buffer_usd(self._session_start_balance)
        chunk            = chunk_risk_usd(self._session_start_balance)

        # ── Adopt any positions already open from a prior run ────────────
        # This prevents: (a) firing a new entry on top of an existing one,
        # and (b) the candle-dedup being reset to 0 so the stale setup fires.
        existing = [
            p for p in (mt5.positions_get(symbol=self.symbol) or [])
            if int(p.magic) == MAGIC
        ]
        if existing:
            log.warning(
                "Found %d pre-existing position(s) from a prior run: %s – adopting them.",
                len(existing),
                [int(p.ticket) for p in existing],
            )
            for p in existing:
                self._known_tickets.add(int(p.ticket))
                # Seed the position manager so trailing/early-exit logic applies
                mp = ManagedPos(
                    ticket    = int(p.ticket),
                    symbol    = self.symbol,
                    direction = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                    entry     = float(p.price_open),
                    sl        = float(p.sl),
                    tp        = float(p.tp),
                    volume    = float(p.volume),
                    risk_usd  = chunk,
                    r_dist    = abs(float(p.price_open) - float(p.sl)) if p.sl else chunk,
                )
                self.pm.positions[mp.ticket] = mp

        log.info("═" * 60)
        log.info(
            "SESSION START │ account=%s │ balance=$%.2f",
            ai.login, self._session_start_balance,
        )
        log.info(
            "BUFFER=$%.2f (%.0f%%) │ chunks=%d │ risk/trade=$%.2f",
            self._buffer_usd, CFG["buffer_pct"] * 100,
            CFG["capital_chunks"], chunk,
        )
        log.info(
            "WITHDRAW when P&L ≥ +$%.2f  │  STOP when P&L ≤ -$%.2f",
            self._buffer_usd, self._buffer_usd,
        )
        log.info("Symbol: %s", self.symbol)
        log.info(
            "AGGRESSION=%d/10 │ conf≥%.2f │ min_rr=%.1f │ drift≤%.4f%% │ "
            "cooldown=%ds │ neutral_h1=%s",
            AGGRESSION,
            CFG["min_confidence"],
            CFG["min_rr"],
            CFG["max_entry_drift_pct"] * 100,
            CFG["loss_cooldown_s"],
            "YES" if CFG["allow_neutral_h1"] else "NO",
        )
        log.info(
            "ENTRY MODES: (1) sweep+reversal  (2) M5 bullish→BUY  (3) M5 bearish→SELL │ "
            "scan every tick │ fill within %ds of %s close │ trend_tp=%.1fR",
            CFG["entry_window_after_close_s"],
            CFG["entry_tf"],
            CFG["trend_tp_r"],
        )
        log.info("═" * 60)
        return True

    # ── session P&L ───────────────────────────────────────────────────────
    def _refresh_pnl(self):
        """
        Session P&L = REALIZED only (broker balance delta since session start).

        Floating losses on open positions are NOT counted here — the trailing
        stop system protects those in real time.  Counting floating would cause
        the engine to stop mid-trade on a normal pullback, before the SL is
        actually hit.  Limits only trigger when real money leaves the account.
        """
        ai = mt5.account_info()
        if ai is None:
            return
        self._session_pnl = float(ai.balance) - self._session_start_balance

    def _check_session_limits(self) -> bool:
        """
        Returns True if trading is allowed.
        Transitions state machine and prints alerts.
        """
        self._refresh_pnl()

        if self._state == "STOPPED":
            return False

        if self._state == "WITHDRAW":
            # Keep repeating the alert every 30 s until human closes the engine
            log.warning(
                "▶▶▶ WITHDRAW ALERT ◀◀◀  Session P&L = +$%.2f  "
                "Withdraw $%.2f to bank then restart the engine.",
                self._session_pnl, self._buffer_usd,
            )
            return False

        # ── Check profit target ────────────────────────────────────────────
        if self._session_pnl >= self._buffer_usd:
            self._state = "WITHDRAW"
            log.warning("=" * 60)
            log.warning("PROFIT TARGET HIT  │  Session P&L = +$%.2f", self._session_pnl)
            log.warning("ACTION REQUIRED: Withdraw $%.2f to your bank account.", self._buffer_usd)
            log.warning("Engine is now PAUSED. Restart after withdrawal.")
            log.warning("=" * 60)
            return False

        # ── Check daily loss limit ─────────────────────────────────────────
        if self._session_pnl <= -self._buffer_usd:
            self._state = "STOPPED"
            log.error("=" * 60)
            log.error("DAILY LOSS LIMIT HIT  │  Session P&L = -$%.2f", abs(self._session_pnl))
            log.error("Buffer of $%.2f is exhausted. NO MORE TRADES TODAY.", self._buffer_usd)
            log.error("Restart the engine tomorrow.")
            log.error("=" * 60)
            return False

        return True

    # ── guards ────────────────────────────────────────────────────────────
    def _in_cooldown(self) -> bool:
        return (time.time() - self._last_close_ts) < CFG["loss_cooldown_s"]

    def _count_open(self) -> int:
        positions = mt5.positions_get(symbol=self.symbol) or []
        return sum(1 for p in positions if int(p.magic) == MAGIC)

    def _sync_entry_window(self, now: float) -> tuple[bool, bool]:
        """
        Update window when a new M5 bar closes.
        Returns (in_window, just_opened).
        """
        closed_ts = last_closed_bar_ts(self.symbol)
        if closed_ts is None:
            return False, False

        just_opened = False
        if closed_ts != self._last_closed_candle_ts:
            self._last_closed_candle_ts = closed_ts
            self._entry_window_until = now + CFG["entry_window_after_close_s"]
            just_opened = True
            log.info(
                "  [M5 CLOSE] New %s bar │ ts=%d │ entry window OPEN for %ds │ "
                "checking: sweep+reversal │ M5 up │ M5 down",
                CFG["entry_tf"], closed_ts, CFG["entry_window_after_close_s"],
            )

        in_window = now < self._entry_window_until
        return in_window, just_opened

    def _detect_new_closes(self):
        live = {
            int(p.ticket)
            for p in (mt5.positions_get(symbol=self.symbol) or ())
            if int(p.magic) == MAGIC
        }
        gone = self._known_tickets - live
        if gone:
            self._last_close_ts = time.time()
            # Allow a new setup on the same M5 bar after the previous trade closed
            self._last_order_candle_ts = 0
            log.info(
                "Position(s) %s closed. Cooldown %ds. Can re-enter on new setup. P&L ≈ $%.2f",
                gone, CFG["loss_cooldown_s"], self._session_pnl,
            )
        self._known_tickets = live

    # ── main loop ─────────────────────────────────────────────────────────
    async def run(self):
        if not self._init():
            return
        self._last_heartbeat: float = 0.0

        while True:
            try:
                await self._tick()
            except Exception:
                log.exception("Unexpected error in main tick")
            await asyncio.sleep(CFG["loop_sleep"])

    async def _tick(self):
        now = time.time()

        # ── 1. Sync broker state ─────────────────────────────────────────
        self._detect_new_closes()
        self.pm.sync(self.symbol)

        # ── 2. Manage existing positions ─────────────────────────────────
        self.pm.update_all(self.symbol)

        # ── 3. Session limit check ────────────────────────────────────────
        if not self._check_session_limits():
            return

        # ── 4. Heartbeat (printed every HEARTBEAT_EVERY seconds) ─────────
        tick = mt5.symbol_info_tick(self.symbol)
        price = ((tick.ask + tick.bid) / 2) if tick else 0.0
        open_count = self._count_open()

        in_window, _ = self._sync_entry_window(now)
        win_left = max(0.0, self._entry_window_until - now)
        next_close = seconds_until_bar_close(self.symbol)

        if now - self._last_heartbeat >= HEARTBEAT_EVERY:
            self._last_heartbeat = now
            cd_left = max(0.0, CFG["loss_cooldown_s"] - (now - self._last_close_ts))
            log.info(
                "── TICK │ price=%.2f │ H1=%s │ state=%s │ agg=%d │ "
                "pnl=$%.2f/buf=$%.2f │ open=%d │ entry_win=%.0fs │ next_M5=%ds ──",
                price,
                h1_trend(self.symbol),
                self._state,
                AGGRESSION,
                self._session_pnl,
                self._buffer_usd,
                open_count,
                win_left,
                next_close,
            )

        # ── 5. Entry guards ──────────────────────────────────────────────
        if self._in_cooldown():
            cd_left = CFG["loss_cooldown_s"] - (now - self._last_close_ts)
            log.debug("  [SKIP] Cooldown active – %.0f s remaining", cd_left)
            return

        if open_count >= CFG["max_open_trades"]:
            log.debug("  [SKIP] %d/%d trade slots in use", open_count, CFG["max_open_trades"])
            return

        # ── 6. Always scan + detect (every tick) ─────────────────────────
        log.debug("  [SCAN] Fetching liquidity levels ...")
        levels = aggregate_levels(self.symbol)
        if not levels:
            log.info("  [SCAN] No liquidity levels found on any TF")
            return

        unswept = [lv for lv in levels if not lv.is_swept]
        scan_line = (
            f"  [SCAN] {len(levels)} levels │ {len(unswept)} unswept │ "
            + "  ".join(f"{lv.ltype}@{lv.price:.1f}(c={lv.confidence:.2f})" for lv in levels[:4])
            + (f" │ entry_window={win_left:.0f}s" if in_window else f" │ next_M5_close={next_close}s")
        )
        if in_window:
            log.info(scan_line)
        else:
            log.debug(scan_line)

        setup = detect_setup(
            self.symbol, levels,
            self._session_start_balance,
            self._last_order_candle_ts,
            quiet=not in_window,
        )
        if setup is None:
            return

        # ── 7. Place order only inside post-close window (tight SL timing) ─
        if not in_window:
            log.debug(
                "  [M5] Setup %s %s valid – fill only after next close "
                "(~%ds then %ds window)",
                setup.entry_mode, setup.direction,
                next_close, CFG["entry_window_after_close_s"],
            )
            return

        log.info(
            "  [ENTRY] %s setup │ %s │ entry≈%.2f │ sl=%.2f │ tp=%.2f │ "
            "rr=%.1f │ window=%.0fs left │ placing MARKET ...",
            setup.entry_mode.upper(), setup.direction, setup.entry, setup.sl, setup.tp,
            setup.rr, win_left,
        )
        ticket = place_order(self.symbol, setup)
        if ticket is None:
            log.warning("  [ORDER] Market order FAILED – will retry next candle")
            return

        df = get_candles(self.symbol, CFG["entry_tf"], 3)
        if df is not None and len(df) >= 2:
            self._last_order_candle_ts = int(df.iloc[-2]["time"])

        self.pm.add(ticket, setup, self.symbol)
        self._known_tickets.add(ticket)
        log.info(
            "  [ORDER] ✓ Filled │ ticket=%d │ %s │ entry=%.5f │ "
            "sl=%.5f │ tp=%.5f │ vol=%.3f │ rr=%.1f │ session_pnl=$%.2f",
            ticket, setup.direction, setup.entry,
            setup.sl, setup.tp, setup.volume, setup.rr, self._session_pnl,
        )


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    engine = TradingEngine(reuse_mt5=False)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        log.info("Shutdown requested by user.")
    finally:
        shutdown_mt5_if_owned(engine._we_initialized_mt5)
        log.info("MT5 disconnected. Engine stopped.")
