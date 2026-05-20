"""
Liquidity sweep + M5 reversal confirmation (matches lib/reversal-chart-overlays.ts).
Supports live prediction on the forming M5 bar built from M1 stream.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

M5_PERIOD_SEC = 300
REVERSAL_CONFIRM_MAX_BARS = 6
REVERSAL_CONFIRM_TF = "M5"
FIXED_RR = 4


class Candle(TypedDict):
    time: int
    open: float
    high: float
    low: float
    close: float


class LiquidityLevel(TypedDict, total=False):
    type: Literal["BSL", "SSL"]
    price: float
    start_time: int
    end_time: int
    is_grabbed: bool


class PendingSetup(TypedDict):
    side: Literal["BUY", "SELL"]
    liquidityTargetPrice: float
    levelType: Literal["BSL", "SSL"]
    sweepTime: int
    sweepCandle: Candle


class ReversalSignal(TypedDict, total=False):
    side: Literal["BUY", "SELL"]
    liquidityPrice: float
    levelType: Literal["BSL", "SSL"]
    sweepTime: int
    reversalTime: int
    entryPrice: float
    sl: float
    tp: float
    sweepCandle: Candle
    reversalCandle: Candle
    isPrediction: bool


def get_reversal_confirm_timeframe(chart_timeframe: str) -> str:
    tf = chart_timeframe.upper()
    return REVERSAL_CONFIRM_TF if tf == "M1" else tf


def uses_separate_confirm_timeframe(chart_timeframe: str) -> bool:
    return get_reversal_confirm_timeframe(chart_timeframe) != chart_timeframe.upper()


def is_bullish_reversal_candle(c: Candle) -> bool:
    return c["close"] > c["open"] and c["low"] < c["open"]


def is_bearish_reversal_candle(c: Candle) -> bool:
    return c["close"] < c["open"] and c["high"] > c["open"]


def count_confirm_bars_since(
    sweep_time: int, confirm_bar_time: int, period_sec: int = M5_PERIOD_SEC
) -> int:
    sweep_bucket = (sweep_time // period_sec) * period_sec
    confirm_bucket = (confirm_bar_time // period_sec) * period_sec
    if confirm_bucket <= sweep_bucket:
        return 0
    return (confirm_bucket - sweep_bucket) // period_sec


def build_forming_m5_from_m1(m1_candles: list[Candle]) -> Optional[Candle]:
    if not m1_candles:
        return None
    last = m1_candles[-1]
    bucket = (last["time"] // M5_PERIOD_SEC) * M5_PERIOD_SEC
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


def _level_key(level_type: str, price: float) -> str:
    return f"{level_type}-{price:.2f}"


def emit_reversal_signal(p: PendingSetup, confirm_bar: Candle, *, is_prediction: bool = False) -> ReversalSignal:
    sweep = p["sweepCandle"]
    liq = p["liquidityTargetPrice"]
    side = p["side"]

    if side == "BUY":
        sl = min(sweep["low"], liq * 0.9999)
        risk = liq - sl
        tp = liq + risk * FIXED_RR
    else:
        sl = max(sweep["high"], liq * 1.0001)
        risk = sl - liq
        tp = liq - risk * FIXED_RR

    return {
        "side": side,
        "liquidityPrice": liq,
        "levelType": p["levelType"],
        "sweepTime": p["sweepTime"],
        "reversalTime": confirm_bar["time"],
        "entryPrice": confirm_bar["close"],
        "sl": sl,
        "tp": tp,
        "sweepCandle": dict(sweep),
        "reversalCandle": dict(confirm_bar),
        "isPrediction": is_prediction,
    }


def can_enter_after_sweep(
    sweep_time: int,
    evaluation_time: int,
    predicted: bool,
    period_sec: int = M5_PERIOD_SEC,
) -> bool:
    if evaluation_time < sweep_time:
        return False
    bars_since = count_confirm_bars_since(sweep_time, evaluation_time, period_sec)
    if bars_since > REVERSAL_CONFIRM_MAX_BARS:
        return False
    if predicted:
        return True
    return bars_since > 0


def scan_reversal_predictions(
    pending: list[PendingSetup],
    forming_confirm: Optional[Candle],
    consumed_keys: Optional[set[str]] = None,
    evaluation_time: Optional[int] = None,
) -> list[ReversalSignal]:
    if not forming_confirm:
        return []
    consumed = consumed_keys or set()
    eval_time = evaluation_time if evaluation_time is not None else forming_confirm["time"]
    out: list[ReversalSignal] = []

    for p in pending:
        if not can_enter_after_sweep(p["sweepTime"], eval_time, True):
            continue
        key = _level_key(p["levelType"], p["liquidityTargetPrice"])
        if key in consumed:
            continue
        buy_ok = p["side"] == "BUY" and is_bullish_reversal_candle(forming_confirm)
        sell_ok = p["side"] == "SELL" and is_bearish_reversal_candle(forming_confirm)
        if not buy_ok and not sell_ok:
            continue
        out.append(emit_reversal_signal(p, forming_confirm, is_prediction=True))

    return out


def scan_reversal_signals_dual(
    sweep_candles: list[Candle],
    confirm_candles: list[Candle],
    liquidity: list[LiquidityLevel],
    *,
    trend_filter: bool = False,
    confirm_period_sec: int = M5_PERIOD_SEC,
) -> list[ReversalSignal]:
    pending: dict[str, PendingSetup] = {}
    consumed: set[str] = set()
    signals: list[ReversalSignal] = []

    for i, current in enumerate(sweep_candles):
        visible = sweep_candles[: i + 1]
        bias: Literal["BULLISH", "BEARISH"] = "BULLISH"
        if trend_filter and len(visible) > 20:
            avg = sum(c["close"] for c in visible[-20:]) / 20
            bias = "BULLISH" if current["close"] > avg else "BEARISH"

        for lvl in liquidity:
            price = float(lvl["price"])
            key = f"{lvl['type']}|{price}"
            if key in consumed or key in pending:
                continue

            if lvl["type"] == "BSL" and current["high"] > price and current["open"] <= price:
                if trend_filter and bias != "BEARISH":
                    continue
                pending[key] = {
                    "side": "SELL",
                    "liquidityTargetPrice": price,
                    "levelType": "BSL",
                    "sweepTime": current["time"],
                    "sweepCandle": dict(current),
                }
            elif lvl["type"] == "SSL" and current["low"] < price and current["open"] >= price:
                if trend_filter and bias != "BULLISH":
                    continue
                pending[key] = {
                    "side": "BUY",
                    "liquidityTargetPrice": price,
                    "levelType": "SSL",
                    "sweepTime": current["time"],
                    "sweepCandle": dict(current),
                }

    for confirm_bar in confirm_candles:
        for key, p in list(pending.items()):
            if confirm_bar["time"] <= p["sweepTime"]:
                continue
            bars_since = count_confirm_bars_since(
                p["sweepTime"], confirm_bar["time"], confirm_period_sec
            )
            if bars_since == 0:
                continue
            if bars_since > REVERSAL_CONFIRM_MAX_BARS:
                del pending[key]
                continue
            buy_ok = p["side"] == "BUY" and is_bullish_reversal_candle(confirm_bar)
            sell_ok = p["side"] == "SELL" and is_bearish_reversal_candle(confirm_bar)
            if not buy_ok and not sell_ok:
                continue
            signals.append(emit_reversal_signal(p, confirm_bar))
            consumed.add(key)
            del pending[key]

    return signals


def detect_pending_setups_from_candle(
    active_candle: Candle,
    liquidity: list[LiquidityLevel],
    *,
    trend_filter: bool = False,
    recent_candles: Optional[list[Candle]] = None,
) -> list[PendingSetup]:
    """Pending sweeps on the latest (forming) chart bar vs liquidity levels."""
    pending: list[PendingSetup] = []
    bias: Literal["BULLISH", "BEARISH"] = "BULLISH"
    if trend_filter and recent_candles and len(recent_candles) > 20:
        avg = sum(c["close"] for c in recent_candles[-20:]) / 20
        bias = "BULLISH" if active_candle["close"] > avg else "BEARISH"

    for lvl in liquidity:
        price = float(lvl["price"])
        if lvl["type"] == "BSL" and active_candle["high"] > price and active_candle["open"] <= price:
            if trend_filter and bias != "BEARISH":
                continue
            pending.append(
                {
                    "side": "SELL",
                    "liquidityTargetPrice": price,
                    "levelType": "BSL",
                    "sweepTime": active_candle["time"],
                    "sweepCandle": dict(active_candle),
                }
            )
        elif lvl["type"] == "SSL" and active_candle["low"] < price and active_candle["open"] >= price:
            if trend_filter and bias != "BULLISH":
                continue
            pending.append(
                {
                    "side": "BUY",
                    "liquidityTargetPrice": price,
                    "levelType": "SSL",
                    "sweepTime": active_candle["time"],
                    "sweepCandle": dict(active_candle),
                }
            )
    return pending


def df_to_candles(df) -> list[Candle]:
    return [
        {
            "time": int(row["time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for _, row in df.iterrows()
    ]


def analyze_live_reversals(
    chart_candles: list[Candle],
    confirm_candles: list[Candle],
    liquidity: list[LiquidityLevel],
    *,
    chart_timeframe: str = "M1",
    trend_filter: bool = False,
    extra_pending: Optional[list[PendingSetup]] = None,
) -> dict[str, Any]:
    """Confirmed signals + forming-bar predictions for live / API consumers."""
    signals = scan_reversal_signals_dual(
        chart_candles,
        confirm_candles,
        liquidity,
        trend_filter=trend_filter,
    )

    pending = list(extra_pending or [])
    if chart_candles:
        active = chart_candles[-1]
        pending.extend(
            detect_pending_setups_from_candle(
                active, liquidity, trend_filter=trend_filter, recent_candles=chart_candles
            )
        )

    forming: Optional[Candle] = None
    if uses_separate_confirm_timeframe(chart_timeframe):
        forming = build_forming_m5_from_m1(chart_candles)
    elif chart_candles:
        forming = chart_candles[-1]

    eval_time = chart_candles[-1]["time"] if chart_candles else 0
    predictions = scan_reversal_predictions(pending, forming, evaluation_time=eval_time)
    confirmed_keys = {
        _level_key(s["levelType"], s["liquidityPrice"])
        for s in signals
        if not s.get("isPrediction")
    }
    predictions = [
        p
        for p in predictions
        if _level_key(p["levelType"], p["liquidityPrice"]) not in confirmed_keys
    ]

    return {
        "chartTimeframe": chart_timeframe.upper(),
        "confirmTimeframe": get_reversal_confirm_timeframe(chart_timeframe),
        "signals": signals,
        "predictions": predictions,
        "formingConfirmCandle": forming,
        "pendingCount": len(pending),
    }
