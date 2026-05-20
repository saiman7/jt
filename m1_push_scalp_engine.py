"""M1 opening-push scalp — sellers open → buy fade; buyers open → sell fade."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def scalp_distances_for_symbol(symbol: str) -> Dict[str, float]:
    s = (symbol or "").upper()
    if "XAU" in s or "GOLD" in s:
        return {"min_push": 0.6, "sl": 1.8, "tp": 1.0}
    if "BTC" in s:
        return {"min_push": 25.0, "sl": 80.0, "tp": 45.0}
    if "USD" in s:
        return {"min_push": 0.00012, "sl": 0.00035, "tp": 0.0002}
    return {"min_push": 0.5, "sl": 1.5, "tp": 0.9}


def detect_opening_push(
    open_: float, high: float, low: float, min_push: float
) -> str:
    up = high - open_
    down = open_ - low
    if down >= min_push and down > up * 1.08:
        return "SELLERS"
    if up >= min_push and up > down * 1.08:
        return "BUYERS"
    return "NONE"


def trade_side_after_push(push: str) -> Optional[str]:
    if push == "SELLERS":
        return "BUY"
    if push == "BUYERS":
        return "SELL"
    return None


def analyze_m1_push_scalp(
    candles: List[dict],
    symbol: str,
    now_sec: Optional[int] = None,
    min_secs_into_bar: int = 8,
) -> Dict[str, Any]:
    import time

    now = int(now_sec if now_sec is not None else time.time())
    dist = scalp_distances_for_symbol(symbol)
    active = candles[-1] if candles else None
    if not active:
        return {"push": "NONE", "signalSide": None, "ready": False}

    bar_time = int(active["time"])
    secs_in = max(0, now - bar_time)
    push = detect_opening_push(
        float(active["open"]),
        float(active["high"]),
        float(active["low"]),
        float(dist["min_push"]),
    )
    side = trade_side_after_push(push)
    ready = secs_in >= min_secs_into_bar and side is not None

    return {
        "push": push,
        "signalSide": side,
        "ready": ready,
        "barTime": bar_time,
        "secondsIntoBar": secs_in,
        "distances": dist,
        "activeCandle": active,
    }
