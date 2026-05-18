"""Higher-timeframe cascade after M5 reversal (M15 → H1 → H4 → D1)."""
from __future__ import annotations

from typing import Any, Literal, Optional

HTF_CASCADE = ("M15", "H1", "H4", "D1")
TF_PERIOD_SEC = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}


def candle_bias(c: dict) -> str:
    if c["close"] > c["open"]:
        return "BULLISH"
    if c["close"] < c["open"]:
        return "BEARISH"
    return "NEUTRAL"


def htf_conflicts_side(side: str, bias: str) -> bool:
    if bias == "NEUTRAL":
        return False
    return (side == "BUY" and bias == "BEARISH") or (side == "SELL" and bias == "BULLISH")


def bucket_start(time: int, period_sec: int) -> int:
    return (time // period_sec) * period_sec


def build_forming_from_m1(m1_candles: list[dict], period_sec: int) -> Optional[dict]:
    if not m1_candles:
        return None
    last = m1_candles[-1]
    bucket = bucket_start(last["time"], period_sec)
    in_bucket = [c for c in m1_candles if c["time"] >= bucket]
    if not in_bucket:
        return None
    return {
        "time": bucket,
        "open": in_bucket[0]["open"],
        "high": max(c["high"] for c in in_bucket),
        "low": min(c["low"] for c in in_bucket),
        "close": in_bucket[-1]["close"],
    }


def last_closed_htf(htf_candles: list[dict], evaluation_time: int, period_sec: int) -> Optional[dict]:
    if len(htf_candles) < 2:
        return None
    forming_bucket = bucket_start(evaluation_time, period_sec)
    closed = [c for c in htf_candles if c["time"] < forming_bucket]
    if closed:
        return closed[-1]
    return htf_candles[-2] if len(htf_candles) >= 2 else None


def evaluate_htf_cascade(
    side: Literal["BUY", "SELL"],
    evaluation_time: int,
    m1_candles: list[dict],
    htf_by_tf: dict[str, list[dict]],
) -> dict[str, Any]:
    checks = []
    path_parts = ["M5✓"]

    for tf in HTF_CASCADE:
        period = TF_PERIOD_SEC[tf]
        series = htf_by_tf.get(tf) or []
        forming = build_forming_from_m1(m1_candles, period) or (series[-1] if series else None)
        closed = last_closed_htf(series, evaluation_time, period)

        if not forming:
            checks.append({"tf": tf, "bias": "NEUTRAL", "aligned": True, "action": "pass"})
            path_parts.append(f"{tf}~")
            continue

        forming_bias = candle_bias(forming)
        bucket_end = forming["time"] + period
        still_forming = evaluation_time < bucket_end

        if not htf_conflicts_side(side, forming_bias):
            checks.append(
                {
                    "tf": tf,
                    "bias": forming_bias,
                    "forming": still_forming,
                    "aligned": True,
                    "action": "pass",
                }
            )
            path_parts.append(f"{tf}✓")
            continue

        if still_forming:
            checks.append(
                {
                    "tf": tf,
                    "bias": forming_bias,
                    "forming": True,
                    "aligned": False,
                    "action": "wait",
                }
            )
            path_parts.append(f"{tf}⏳")
            return {
                "canEnter": False,
                "waitingTf": tf,
                "message": f"{tf} {forming_bias.lower()} — wait for candle close",
                "path": "→".join(path_parts),
                "checks": checks,
            }

        closed_bias = candle_bias(closed) if closed else forming_bias
        if not htf_conflicts_side(side, closed_bias):
            checks.append(
                {
                    "tf": tf,
                    "bias": closed_bias,
                    "forming": False,
                    "aligned": True,
                    "action": "pass",
                }
            )
            path_parts.append(f"{tf}✓")
            continue

        checks.append(
            {
                "tf": tf,
                "bias": closed_bias,
                "forming": False,
                "aligned": False,
                "action": "escalate",
            }
        )
        path_parts.append(f"{tf}↑")

    return {
        "canEnter": True,
        "waitingTf": None,
        "message": "HTF stack aligned",
        "path": "→".join(path_parts),
        "checks": checks,
    }
