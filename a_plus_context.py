"""
A+ trade setup context engine.

Implements the three-question framework from A+setup.md:
  1. HTF momentum from CLOSE-confirmed Break of Structure (wick-only = sweep, not BOS)
  2. MTF zone: premium / discount / equilibrium via Fib on the external range
  3. Side alignment: BUYs only with bullish HTF in discount/equilibrium;
                     SELLs only with bearish HTF in premium/equilibrium.

Strictly used by the live agent to reject sweeps that do not satisfy A+.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

Bias = Literal["BULLISH", "BEARISH", "NEUTRAL"]
Zone = Literal["PREMIUM", "DISCOUNT", "EQUILIBRIUM", "UNKNOWN"]


class Candle(TypedDict):
    time: int
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Symbol-specific A+ defaults. Gold (XAUUSD) is the doc's reference example.
# ---------------------------------------------------------------------------
DEFAULT_PROFILE: Dict[str, Any] = {
    "htf": "H1",
    "mtf": "M15",
    "htf_count": 240,           # ~10 days of H1
    "mtf_count": 192,           # ~48 hours of M15
    "htf_swing_window": 3,
    "mtf_swing_window": 3,
    "mtf_range_lookback": 96,   # last 24h of M15 defines the external range
    "equilibrium_band_pct": 0.05,
}

# Gold-tuned values match the doc (H1 → M15 → M5 stack).
SYMBOL_PROFILES: Dict[str, Dict[str, Any]] = {
    "XAUUSD": dict(DEFAULT_PROFILE),
    "GOLD":   dict(DEFAULT_PROFILE),
}


def profile_for_symbol(symbol: str) -> Dict[str, Any]:
    s = (symbol or "").upper()
    for key, profile in SYMBOL_PROFILES.items():
        if key in s:
            return dict(profile)
    return dict(DEFAULT_PROFILE)


# ---------------------------------------------------------------------------
# Swing pivots (fractal)
# ---------------------------------------------------------------------------
def _find_swing_highs(candles: List[Candle], window: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(candles)
    for i in range(window, n - window):
        h = candles[i]["high"]
        if all(candles[i - k]["high"] < h for k in range(1, window + 1)) and \
           all(candles[i + k]["high"] <= h for k in range(1, window + 1)):
            out.append({"idx": i, "time": int(candles[i]["time"]), "price": float(h)})
    return out


def _find_swing_lows(candles: List[Candle], window: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(candles)
    for i in range(window, n - window):
        l = candles[i]["low"]
        if all(candles[i - k]["low"] > l for k in range(1, window + 1)) and \
           all(candles[i + k]["low"] >= l for k in range(1, window + 1)):
            out.append({"idx": i, "time": int(candles[i]["time"]), "price": float(l)})
    return out


# ---------------------------------------------------------------------------
# 1) HTF bias from close-confirmed BOS
# ---------------------------------------------------------------------------
def compute_htf_bias(
    candles: List[Candle],
    *,
    swing_window: int = 3,
) -> Dict[str, Any]:
    """
    Doc rule: "If the candle didn't close the high, it's not a valid break of
    structure. It's just a sweep. So we only look for a candle closure."

    We drop the still-forming bar (we cannot trust its close), then find every
    swing high that was later closed THROUGH (BULLISH BOS) and every swing low
    that was closed BELOW (BEARISH BOS). The most recent of these events sets
    the bias.
    """
    closed = candles[:-1] if len(candles) >= 2 else candles
    n = len(closed)
    if n < swing_window * 4 + 5:
        return {"bias": "NEUTRAL", "reason": "Insufficient closed HTF bars"}

    highs = _find_swing_highs(closed, swing_window)
    lows = _find_swing_lows(closed, swing_window)

    events: List[Dict[str, Any]] = []

    for sh in highs:
        confirm_from = sh["idx"] + swing_window + 1
        for j in range(confirm_from, n):
            if closed[j]["close"] > sh["price"]:
                events.append({
                    "time": int(closed[j]["time"]),
                    "bias": "BULLISH",
                    "swing": sh["price"],
                    "close": float(closed[j]["close"]),
                })
                break

    for sl in lows:
        confirm_from = sl["idx"] + swing_window + 1
        for j in range(confirm_from, n):
            if closed[j]["close"] < sl["price"]:
                events.append({
                    "time": int(closed[j]["time"]),
                    "bias": "BEARISH",
                    "swing": sl["price"],
                    "close": float(closed[j]["close"]),
                })
                break

    swing_high_prices = [h["price"] for h in highs[-3:]]
    swing_low_prices = [l["price"] for l in lows[-3:]]

    if not events:
        return {
            "bias": "NEUTRAL",
            "reason": "No close-based BOS in lookback",
            "swingHighs": swing_high_prices,
            "swingLows": swing_low_prices,
        }

    events.sort(key=lambda e: e["time"])
    latest = events[-1]
    return {
        "bias": latest["bias"],
        "reason": f"BOS by close past swing {latest['swing']:.2f}",
        "bosTime": latest["time"],
        "bosSwingPrice": latest["swing"],
        "bosCloseAt": latest["close"],
        "swingHighs": swing_high_prices,
        "swingLows": swing_low_prices,
    }


# ---------------------------------------------------------------------------
# 2) MTF zone via Fib on external range
# ---------------------------------------------------------------------------
def compute_zone(
    mtf_candles: List[Candle],
    current_price: float,
    *,
    swing_window: int = 3,
    range_lookback: int = 96,
    equilibrium_band_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    External range = highest high & lowest low over the recent MTF lookback.
    Premium  = above midpoint + band   (sell zone in a bearish leg)
    Discount = below midpoint - band   (buy zone in a bullish leg)
    Equilibrium = within the band around the midpoint (50% fib).
    """
    closed = mtf_candles[:-1] if len(mtf_candles) >= 2 else mtf_candles
    if len(closed) < max(swing_window * 2 + 5, 12):
        return {"zone": "UNKNOWN", "reason": "Insufficient MTF bars"}

    recent = closed[-range_lookback:] if len(closed) > range_lookback else closed
    rh = max(c["high"] for c in recent)
    rl = min(c["low"] for c in recent)
    if rh <= rl:
        return {"zone": "UNKNOWN", "reason": "Range collapsed"}

    mid = (rh + rl) / 2.0
    span = rh - rl
    band = span * equilibrium_band_pct

    if current_price > mid + band:
        zone: Zone = "PREMIUM"
    elif current_price < mid - band:
        zone = "DISCOUNT"
    else:
        zone = "EQUILIBRIUM"

    pct_in_range = (current_price - rl) / span

    return {
        "zone": zone,
        "rangeHigh": float(rh),
        "rangeLow": float(rl),
        "midpoint": float(mid),
        "equilibriumBand": float(band),
        "currentPrice": float(current_price),
        "pctInRange": float(pct_in_range),
        "rangeBars": len(recent),
    }


# ---------------------------------------------------------------------------
# 3) Full context + strict validator
# ---------------------------------------------------------------------------
def compute_a_plus_context(
    htf_candles: List[Candle],
    mtf_candles: List[Candle],
    current_price: float,
    *,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    p = profile or DEFAULT_PROFILE
    bias_info = compute_htf_bias(htf_candles, swing_window=p["htf_swing_window"])
    zone_info = compute_zone(
        mtf_candles,
        current_price,
        swing_window=p["mtf_swing_window"],
        range_lookback=p["mtf_range_lookback"],
        equilibrium_band_pct=p["equilibrium_band_pct"],
    )
    return {
        "htfTimeframe": p["htf"],
        "mtfTimeframe": p["mtf"],
        "htf": bias_info,
        "mtf": zone_info,
    }


def validate_a_plus_entry(
    side: Literal["BUY", "SELL"],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Strict A+ rules from A+setup.md:
      - HTF bias must be clear (no NEUTRAL).
      - Side must align with HTF bias.
      - BUYs only in DISCOUNT or EQUILIBRIUM (never premium).
      - SELLs only in PREMIUM or EQUILIBRIUM (never discount).
      - Equilibrium requires LTF reversal confirmation (already enforced by
        the reversal engine via the rejection-candle pattern).
    """
    if not context:
        return {"allowed": False, "rule": None, "reason": "Missing A+ context"}

    bias = (context.get("htf") or {}).get("bias", "NEUTRAL")
    zone = (context.get("mtf") or {}).get("zone", "UNKNOWN")

    if bias == "NEUTRAL":
        return {"allowed": False, "rule": None, "reason": "HTF bias unclear (no close-based BOS)"}
    if zone == "UNKNOWN":
        return {"allowed": False, "rule": None, "reason": "MTF zone unknown — insufficient structure"}

    if side == "BUY" and bias != "BULLISH":
        return {"allowed": False, "rule": None, "reason": f"BUY against HTF {bias.lower()} bias"}
    if side == "SELL" and bias != "BEARISH":
        return {"allowed": False, "rule": None, "reason": f"SELL against HTF {bias.lower()} bias"}

    if side == "BUY" and zone == "PREMIUM":
        return {"allowed": False, "rule": None, "reason": "BUY in premium — need discount or equilibrium"}
    if side == "SELL" and zone == "DISCOUNT":
        return {"allowed": False, "rule": None, "reason": "SELL in discount — need premium or equilibrium"}

    if side == "BUY" and zone == "DISCOUNT":
        return {"allowed": True, "rule": "A+", "reason": "Bullish HTF + buy in discount"}
    if side == "SELL" and zone == "PREMIUM":
        return {"allowed": True, "rule": "A+", "reason": "Bearish HTF + sell in premium"}

    return {
        "allowed": True,
        "rule": "B-equilibrium",
        "reason": f"{side} at equilibrium — LTF reversal acts as confluence",
    }
