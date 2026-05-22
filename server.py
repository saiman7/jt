import asyncio
import math
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Ensure `jt/` is on sys.path when uvicorn is started outside this folder
_JT_DIR = Path(__file__).resolve().parent
if str(_JT_DIR) not in sys.path:
    sys.path.insert(0, str(_JT_DIR))
from typing import Any, List, Optional, Tuple, TypedDict
import pandas as pd
import MetaTrader5 as mt5
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware

from reversal_engine import (
    analyze_live_reversals,
    build_forming_m5_from_m1,
    df_to_candles,
    get_reversal_confirm_timeframe,
    uses_separate_confirm_timeframe,
)
from m1_push_scalp_engine import analyze_m1_push_scalp

app = FastAPI()

# Enable CORS to allow your frontend application to securely access this backend api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not mt5.initialize():
    print("Failed to initialize MT5")
    mt5.shutdown()

# symbol_info.filling_mode bitmask (MQL5 SYMBOL_FILLING_*; not exported by MetaTrader5 Python)
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

AGENT_MAGIC = 123456
M1_PUSH_SCALP_MAGIC = 123457


def resolve_mt5_symbol(raw: str) -> Optional[str]:
    """
    Return the symbol string exactly as the broker exposes it in MT5.
    UI often passes mixed case (e.g. BTCUSDr) while .upper() breaks symbol_info on some servers.
    """
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    for candidate in (s, s.upper(), s.lower()):
        info = mt5.symbol_info(candidate)
        if info is not None:
            return info.name
    needle = s.upper()
    for sym in mt5.symbols_get() or ():
        if sym.name.upper() == needle:
            return sym.name
    return None


def copy_rates_as_candles(
    symbol: str, timeframe: str, count: int
) -> Tuple[List[dict], Optional[str]]:
    """Fetch last N bars from MT5 as candle dicts."""
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        return [], f"No MT5 symbol matching {symbol!r}."
    tf = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_from_pos(resolved, tf, 0, count)
    if rates is None or len(rates) == 0:
        return [], "No rates"
    df = pd.DataFrame(rates)
    return df_to_candles(df), None


def calculate_advanced_liquidity(df, window=5, source_tag="Local"):
    liquidity_segments = []
    total_candles = len(df)
    seen_level_keys = set()

    for i in range(window, total_candles - window):
        current_high = float(df.iloc[i]['high'])
        current_low = float(df.iloc[i]['low'])
        start_ts = int(df.iloc[i]['time'])

        # 1. Swing High (BSL)
        if current_high > df.iloc[i-window:i]['high'].max() and current_high >= df.iloc[i+1:i+window+1]['high'].max():
            rounded_price = round(current_high, 2)
            bsl_key = f"BSL-{rounded_price}"
            if bsl_key not in seen_level_keys:
                seen_level_keys.add(bsl_key)
                end_ts = int(df.iloc[-1]['time'])
                has_been_swept = False
                
                for j in range(i + window + 1, total_candles):
                    if float(df.iloc[j]['high']) > current_high:
                        end_ts = int(df.iloc[j]['time'])
                        has_been_swept = True
                        break
                liquidity_segments.append({
                    "type": "BSL", 
                    "price": current_high, 
                    "start_time": start_ts, 
                    "end_time": end_ts, 
                    "is_grabbed": has_been_swept,
                    "level_source": source_tag # 💡 Added source tagging to filter targets
                })

        # 2. Swing Low (SSL)
        if current_low < df.iloc[i-window:i]['low'].min() and current_low <= df.iloc[i+1:i+window+1]['low'].min():
            rounded_price = round(current_low, 2)
            ssl_key = f"SSL-{rounded_price}"
            if ssl_key not in seen_level_keys:
                seen_level_keys.add(ssl_key)
                end_ts = int(df.iloc[-1]['time'])
                has_been_swept = False
                
                for j in range(i + window + 1, total_candles):
                    if float(df.iloc[j]['low']) < current_low:
                        end_ts = int(df.iloc[j]['time'])
                        has_been_swept = True
                        break
                liquidity_segments.append({
                    "type": "SSL", 
                    "price": current_low, 
                    "start_time": start_ts, 
                    "end_time": end_ts, 
                    "is_grabbed": has_been_swept,
                    "level_source": source_tag # 💡 Added source tagging to filter targets
                })

    return liquidity_segments


def calculate_fib_golden_zones(swings, max_recent=12, source_prefix="Fib_50"):
    """
    Pair consecutive opposite swings into legs and emit the 50% retracement as a
    sweep target. Output reuses BSL/SSL types so the existing engine treats them
    like structural liquidity (one-candle SL, post-sweep confirm, MT5 sizing).

      Bearish leg  (BSL -> SSL)  → 50% emitted as BSL  (sweep up + reject = SELL)
      Bullish leg  (SSL -> BSL)  → 50% emitted as SSL  (sweep down + reject = BUY)
    """
    if not swings or len(swings) < 2:
        return []

    sorted_swings = sorted(swings, key=lambda s: s.get("start_time", 0))
    zones = []

    for i in range(1, len(sorted_swings)):
        prev = sorted_swings[i - 1]
        curr = sorted_swings[i]
        prev_type = prev.get("type")
        curr_type = curr.get("type")
        if prev_type == curr_type:
            continue

        try:
            prev_price = float(prev.get("price", 0))
            curr_price = float(curr.get("price", 0))
        except (TypeError, ValueError):
            continue

        # Bearish leg: high → low ; mid is a SELL retracement target
        if prev_type == "BSL" and curr_type == "SSL":
            high = prev_price
            low = curr_price
            if high <= low:
                continue
            mid = round((high + low) / 2.0, 5)
            zones.append({
                "type": "BSL",
                "price": mid,
                "start_time": int(curr.get("start_time", 0)),
                "end_time": int(curr.get("end_time", 0)),
                "is_grabbed": False,
                "level_source": source_prefix,
                "leg_high": high,
                "leg_low": low,
            })

        # Bullish leg: low → high ; mid is a BUY retracement target
        elif prev_type == "SSL" and curr_type == "BSL":
            low = prev_price
            high = curr_price
            if high <= low:
                continue
            mid = round((high + low) / 2.0, 5)
            zones.append({
                "type": "SSL",
                "price": mid,
                "start_time": int(curr.get("start_time", 0)),
                "end_time": int(curr.get("end_time", 0)),
                "is_grabbed": False,
                "level_source": source_prefix,
                "leg_high": high,
                "leg_low": low,
            })

    return zones[-max_recent:]


# -------------------------------------------------------------
# UPGRADED ROUTE: MULTI-TIMEFRAME LIQUIDITY AGGREGATOR
# -------------------------------------------------------------
@app.get("/api/historical")
async def get_historical_backtest_data(
    symbol: str = Query("BTCUSDz"), 
    timeframe: str = Query("M5"), 
    start_time: str = Query(None),  # Expects: ISO String (YYYY-MM-DDTHH:MM) or Unix Timestamp
    end_time: str = Query(None)
):
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        return {"error": f"No MT5 symbol matching {symbol!r}. Add it in Market Watch (Show All) and use the exact name."}
    symbol = resolved
    mt5_tf_ltf = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M5)
    mt5_tf_htf = mt5.TIMEFRAME_H4 # 💡 Fixed Higher Timeframe for structural context mapping
    
    if start_time and end_time:
        try:
            if "T" in start_time:
                date_start = datetime.fromisoformat(start_time)
                date_end = datetime.fromisoformat(end_time)
            else:
                date_start = datetime.fromtimestamp(int(start_time))
                date_end = datetime.fromtimestamp(int(end_time))
                
            # Padding conversion to safely ingest trailing contextual data models
            tf_num = int(timeframe.replace("M", "").replace("H", "60").replace("D", "1440")) if any(x in timeframe for x in ["M", "H", "D"]) else 5
            minutes_padding = 100 * tf_num # Double lookback depth protection
            padded_start = datetime.fromtimestamp(int(date_start.timestamp()) - (minutes_padding * 60))

            # 1. Fetch Execution Data (Lower Timeframe)
            rates_ltf = mt5.copy_rates_range(symbol, mt5_tf_ltf, padded_start, date_end)
            
            # 2. Fetch Macro Context Data (H4 Anchor Timeframe)
            rates_htf = mt5.copy_rates_range(symbol, mt5_tf_htf, padded_start, date_end)

        except Exception as err:
            return {"error": f"Date parsing layout breakdown: {str(err)}"}
    else:
        rates_ltf = mt5.copy_rates_from_pos(symbol, mt5_tf_ltf, 0, 1000)
        rates_htf = mt5.copy_rates_from_pos(symbol, mt5_tf_htf, 0, 250)
    
    if rates_ltf is None or len(rates_ltf) == 0:
        return {"error": f"No historical execution records found for symbol: {symbol}"}
        
    df_ltf = pd.DataFrame(rates_ltf)
    
    # Extract lower timeframe liquidity sweeps (Window 5 for structural pivots)
    ltf_liquidity = calculate_advanced_liquidity(df_ltf, window=5, source_tag="Local")
    
    # Process and append H4 Higher Timeframe Liquidity Targets
    htf_liquidity = []
    if rates_htf is not None and len(rates_htf) > 0:
        df_htf = pd.DataFrame(rates_htf)
        # Use a larger structural swing lookback window (e.g., 10) for macro levels
        htf_liquidity = calculate_advanced_liquidity(df_htf, window=10, source_tag="Macro_H4")

    # Combine both local triggers and macro targets into an integrated landscape array
    combined_liquidity = ltf_liquidity + htf_liquidity

    # Add Fib 50% golden-zone retracement targets for trend continuation entries
    fib_local = calculate_fib_golden_zones(ltf_liquidity, max_recent=12, source_prefix="Fib_50_Local")
    fib_macro = calculate_fib_golden_zones(htf_liquidity, max_recent=6, source_prefix="Fib_50_Macro_H4")
    combined_liquidity = combined_liquidity + fib_local + fib_macro

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": df_ltf[["time", "open", "high", "low", "close"]].to_dict(orient="records"),
        "liquidity": combined_liquidity
    }


@app.get("/api/recent-candles")
async def recent_candles(
    symbol: str = Query(...),
    timeframe: str = Query("M1"),
    count: int = Query(16, ge=2, le=500),
):
    """Last N closed+forming bars for sweep confirmation (e.g. M5 when chart is M1)."""
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        return {"error": f"No MT5 symbol matching {symbol!r}.", "candles": []}
    tf = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_from_pos(resolved, tf, 0, count)
    if rates is None or len(rates) == 0:
        return {"error": "No rates", "candles": []}
    df = pd.DataFrame(rates)
    return {
        "symbol": resolved,
        "timeframe": timeframe.upper(),
        "candles": df[["time", "open", "high", "low", "close"]].to_dict(orient="records"),
    }


@app.get("/api/health")
async def health_check():
    """Liveness probe for the MT5 gateway."""
    terminal = mt5.terminal_info()
    account = mt5.account_info()
    return {
        "ok": terminal is not None,
        "mt5_connected": terminal is not None,
        "terminal": terminal.name if terminal else None,
        "account_login": account.login if account else None,
    }


@app.get("/api/account-balance")
async def account_balance():
    """Current MT5 account balance (realized PnL applies on position close)."""
    account = mt5.account_info()
    if account is None:
        return {"success": False, "error": "account_info unavailable", "mt5_error": mt5.last_error()}
    return {
        "success": True,
        "balance": round(float(account.balance), 2),
        "equity": round(float(account.equity), 2),
        "login": int(account.login),
    }


@app.get("/api/forming-candle")
async def forming_candle(
    symbol: str = Query(...),
    chart_timeframe: str = Query("M1"),
    m1_count: int = Query(300, ge=10, le=2000),
):
    """
    Live forming confirm candle (e.g. aggregate current M5 bucket from M1 stream).
    Use before M5 close for reversal prediction.
    """
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        return {"error": f"No MT5 symbol matching {symbol!r}."}

    chart_tf = chart_timeframe.upper()
    if uses_separate_confirm_timeframe(chart_tf):
        m1_candles, err = copy_rates_as_candles(resolved, "M1", m1_count)
        if err:
            return {"error": err, "forming": None}
        forming = build_forming_m5_from_m1(m1_candles)
        return {
            "symbol": resolved,
            "chartTimeframe": chart_tf,
            "confirmTimeframe": get_reversal_confirm_timeframe(chart_tf),
            "forming": forming,
        }

    candles, err = copy_rates_as_candles(resolved, chart_tf, min(m1_count, 500))
    if err:
        return {"error": err, "forming": None}
    forming = candles[-1] if candles else None
    return {
        "symbol": resolved,
        "chartTimeframe": chart_tf,
        "confirmTimeframe": chart_tf,
        "forming": forming,
    }


@app.get("/api/reversal-live")
async def reversal_live(
    symbol: str = Query(...),
    chart_timeframe: str = Query("M1"),
    chart_count: int = Query(300, ge=50, le=2000),
    confirm_count: int = Query(150, ge=20, le=500),
    trend_filter: bool = Query(False),
):
    """
    Confirmed reversal signals + forming-bar predictions from live MT5 data.
    Chart TF M1 → sweeps on M1, confirm/predict on forming M5.
    """
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        return {"error": f"No MT5 symbol matching {symbol!r}."}

    chart_tf = chart_timeframe.upper()
    chart_candles, err = copy_rates_as_candles(resolved, chart_tf, chart_count)
    if err:
        return {"error": err}

    confirm_tf = get_reversal_confirm_timeframe(chart_tf)
    if uses_separate_confirm_timeframe(chart_tf):
        confirm_candles, cerr = copy_rates_as_candles(resolved, confirm_tf, confirm_count)
        if cerr:
            return {"error": cerr}
    else:
        confirm_candles = chart_candles

    df_chart = pd.DataFrame(chart_candles)
    liquidity = calculate_advanced_liquidity(df_chart, window=5, source_tag="Local")
    fib = calculate_fib_golden_zones(liquidity, max_recent=6, source_prefix="Fib_50_Local")
    combined = liquidity + fib

    result = analyze_live_reversals(
        chart_candles,
        confirm_candles,
        combined,
        chart_timeframe=chart_tf,
        trend_filter=trend_filter,
    )
    return {
        "symbol": resolved,
        "liquidity": combined[-20:],
        **result,
    }


class ReversalScanBody(BaseModel):
    chart_candles: List[dict] = Field(..., min_length=1)
    confirm_candles: Optional[List[dict]] = None
    liquidity: List[dict] = Field(default_factory=list)
    chart_timeframe: str = "M1"
    trend_filter: bool = False


@app.post("/api/reversal-scan")
async def reversal_scan(body: ReversalScanBody):
    """Scan client-supplied candles (backtest export or custom feed) for reversals + predictions."""
    chart_tf = body.chart_timeframe.upper()
    confirm = body.confirm_candles
    if confirm is None:
        confirm = body.chart_candles

    chart = [
        {
            "time": int(c["time"]),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
        }
        for c in body.chart_candles
    ]
    confirm_candles = [
        {
            "time": int(c["time"]),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
        }
        for c in confirm
    ]
    liquidity = body.liquidity

    result = analyze_live_reversals(
        chart,
        confirm_candles,
        liquidity,
        chart_timeframe=chart_tf,
        trend_filter=body.trend_filter,
    )
    return result


@app.websocket("/ws/rates")
async def websocket_rates_endpoint(websocket: WebSocket, symbol: str = "BTCUSD", timeframe: str = "M1"):
    await websocket.accept()
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        await websocket.close(code=4404, reason=f"Unknown symbol: {symbol}")
        return
    symbol = resolved
    mt5_tf = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M1)
    
    try:
        while True:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 250)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                chart_candles = df_to_candles(df)
                advanced_liquidity = calculate_advanced_liquidity(df, window=5, source_tag="Local")
                fib_zones = calculate_fib_golden_zones(advanced_liquidity, max_recent=6, source_prefix="Fib_50_Local")
                combined_liquidity = advanced_liquidity[-10:] + fib_zones

                payload: dict[str, Any] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": df[["time", "open", "high", "low", "close"]].to_dict(orient="records"),
                    "liquidity": combined_liquidity,
                }

                chart_tf = timeframe.upper()
                if uses_separate_confirm_timeframe(chart_tf):
                    confirm_candles, _ = copy_rates_as_candles(symbol, get_reversal_confirm_timeframe(chart_tf), 80)
                    reversal = analyze_live_reversals(
                        chart_candles,
                        confirm_candles,
                        combined_liquidity,
                        chart_timeframe=chart_tf,
                        trend_filter=True,
                    )
                    payload["formingConfirmCandle"] = reversal.get("formingConfirmCandle")
                    payload["reversalSignals"] = reversal.get("signals", [])
                    payload["reversalPredictions"] = reversal.get("predictions", [])
                    payload["confirmTimeframe"] = reversal.get("confirmTimeframe")

                await websocket.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/m1-push-scalp")
async def websocket_m1_push_scalp(
    websocket: WebSocket,
    symbol: str = "XAUUSD",
    min_secs_into_bar: int = 8,
):
    """Fast M1 tick stream + opening-push scalp signal (separate from liquidity agent)."""
    await websocket.accept()
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        await websocket.close(code=4404, reason=f"Unknown symbol: {symbol}")
        return
    symbol = resolved

    try:
        while True:
            tick = mt5.symbol_info_tick(symbol)
            candles, err = copy_rates_as_candles(symbol, "M1", 8)
            payload: dict[str, Any] = {
                "symbol": symbol,
                "timeframe": "M1",
                "error": err,
                "data": candles,
                "tick": None,
            }
            if tick is not None:
                payload["tick"] = {
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "time": int(tick.time),
                }
            if candles:
                payload["scalp"] = analyze_m1_push_scalp(
                    candles,
                    symbol,
                    min_secs_into_bar=min_secs_into_bar,
                )
            await websocket.send_json(payload)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass


def normalize_volume(volume: float, symbol_info) -> float:
    """Snap volume down to the symbol's lot step (e.g. 0.208 → 0.20 when step is 0.01)."""
    step = float(symbol_info.volume_step) if symbol_info.volume_step else 0.01
    min_vol = float(symbol_info.volume_min) if symbol_info.volume_min else step
    if step <= 0:
        step = 0.01
    normalized = math.floor(volume / step) * step
    normalized = max(min_vol, normalized)
    step_text = f"{step:.10f}".rstrip("0")
    decimals = len(step_text.split(".")[1]) if "." in step_text else 2
    return round(normalized, decimals)


class LotForRiskIn(BaseModel):
    """Broker-accurate sizing: matches account currency P&L at SL for the given volume."""

    symbol: str
    action: str
    entry_price: float = Field(..., gt=0)
    sl_price: float = Field(..., gt=0)
    risk_usd: float = Field(..., gt=0)


@app.post("/api/lot-for-risk")
async def lot_for_risk(req: LotForRiskIn):
    """Return lot size so that if SL is hit, loss in account currency ≈ risk_usd (before slippage)."""
    resolved = resolve_mt5_symbol(req.symbol)
    if resolved is None:
        return {"success": False, "error": f"Symbol {req.symbol!r} not found in MT5."}

    action_upper = req.action.upper()
    if action_upper not in ("BUY", "SELL"):
        return {"success": False, "error": "action must be BUY or SELL"}

    symbol_info = mt5.symbol_info(resolved)
    if symbol_info is None:
        return {"success": False, "error": "symbol_info unavailable"}

    order_type = mt5.ORDER_TYPE_BUY if action_upper == "BUY" else mt5.ORDER_TYPE_SELL
    entry = float(req.entry_price)
    sl = float(req.sl_price)

    loss_one_lot = mt5.order_calc_profit(order_type, resolved, 1.0, entry, sl)
    if loss_one_lot is None:
        return {"success": False, "error": "order_calc_profit failed (invalid prices or symbol)."}

    loss_mag = abs(float(loss_one_lot))
    if loss_mag < 1e-12:
        return {"success": False, "error": "Zero loss at SL for 1.0 lot — widen stop or check SL side."}

    raw_vol = float(req.risk_usd) / loss_mag
    volume = normalize_volume(raw_vol, symbol_info)

    max_vol = float(symbol_info.volume_max) if symbol_info.volume_max else 1000.0
    if volume > max_vol + 1e-9:
        return {
            "success": False,
            "error": f"Required volume {volume} exceeds symbol maximum {max_vol}.",
        }

    actual = mt5.order_calc_profit(order_type, resolved, volume, entry, sl)
    if actual is None:
        return {"success": False, "error": "order_calc_profit failed for normalized volume."}

    estimated_risk = abs(float(actual))
    return {
        "success": True,
        "volume": volume,
        "loss_per_lot": loss_mag,
        "estimated_risk_usd": round(estimated_risk, 2),
    }


class CalcProfitIn(BaseModel):
    symbol: str
    action: str
    volume: float = Field(..., gt=0)
    price_open: float = Field(..., gt=0)
    price_close: float = Field(..., gt=0)


@app.post("/api/calc-profit")
async def calc_profit_endpoint(req: CalcProfitIn):
    """P&L in account currency for closing volume at price_close (excludes balance ops; may miss some fees vs deals)."""
    resolved = resolve_mt5_symbol(req.symbol)
    if resolved is None:
        return {"success": False, "error": f"Symbol {req.symbol!r} not found in MT5."}

    action_upper = req.action.upper()
    if action_upper not in ("BUY", "SELL"):
        return {"success": False, "error": "action must be BUY or SELL"}

    if not mt5.symbol_select(resolved, True):
        pass  # order_calc_profit may still work

    order_type = mt5.ORDER_TYPE_BUY if action_upper == "BUY" else mt5.ORDER_TYPE_SELL
    profit = mt5.order_calc_profit(
        order_type,
        resolved,
        float(req.volume),
        float(req.price_open),
        float(req.price_close),
    )
    if profit is None:
        return {"success": False, "error": "order_calc_profit failed", "mt5_error": mt5.last_error()}
    return {"success": True, "profit": round(float(profit), 2)}


@app.get("/api/closed-position-pnl")
async def closed_position_pnl(ticket: int = Query(..., gt=0)):
    """Sum profit+commission+swap for all deals on this position ticket; weighted avg exit from OUT deals."""
    utc_now = datetime.now(timezone.utc)
    from_dt = utc_now - timedelta(days=90)
    deals = mt5.history_deals_get(from_dt, utc_now, group="*", position=ticket)
    if deals is None:
        return {"success": False, "error": "history_deals_get failed", "mt5_error": mt5.last_error()}
    if len(deals) == 0:
        return {"success": False, "error": "no deals for position"}

    total = 0.0
    out_deals = []
    for d in deals:
        total += float(d.profit) + float(d.commission) + float(d.swap)
        if d.entry == mt5.DEAL_ENTRY_OUT:
            out_deals.append(d)

    exit_price = None
    if out_deals:
        vol_sum = sum(float(x.volume) for x in out_deals)
        if vol_sum > 1e-12:
            exit_price = sum(float(x.price) * float(x.volume) for x in out_deals) / vol_sum
        else:
            exit_price = float(max(out_deals, key=lambda x: x.time).price)

    return {
        "success": True,
        "total_pnl": round(total, 2),
        "exit_price": round(exit_price, 5) if exit_price is not None else None,
        "deal_count": len(deals),
    }


def _resolve_sl_tp_prices(
    action_upper: str,
    fill_price: float,
    symbol_point: float,
    sl: Optional[float],
    tp: Optional[float],
    sl_points: Optional[int],
    tp_points: Optional[int],
) -> Tuple[float, float]:
    """Prefer absolute sl/tp from the client; fall back to point offsets from fill price."""
    if action_upper == "BUY":
        sl_price = float(sl) if sl and sl > 0 else (
            fill_price - (sl_points * symbol_point) if sl_points else 0.0
        )
        tp_price = float(tp) if tp and tp > 0 else (
            fill_price + (tp_points * symbol_point) if tp_points else 0.0
        )
    else:
        sl_price = float(sl) if sl and sl > 0 else (
            fill_price + (sl_points * symbol_point) if sl_points else 0.0
        )
        tp_price = float(tp) if tp and tp > 0 else (
            fill_price - (tp_points * symbol_point) if tp_points else 0.0
        )
    return sl_price, tp_price


@app.get("/api/positions")
async def get_open_positions(
    symbol: str = Query(None),
    magic: int = Query(AGENT_MAGIC),
):
    """Return agent-managed open positions so the frontend never stacks orders."""
    if symbol:
        resolved = resolve_mt5_symbol(symbol)
        if resolved is None:
            return []
        positions = mt5.positions_get(symbol=resolved)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return {"positions": []}

    filtered = [p for p in positions if int(p.magic) == int(magic)]
    return {
        "positions": [
            {
                "ticket": int(p.ticket),
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": float(p.volume),
                "price_open": float(p.price_open),
                "sl": float(p.sl),
                "tp": float(p.tp),
                "profit": float(p.profit),
                "time": int(p.time),
            }
            for p in filtered
        ]
    }


@app.post("/api/trade")
async def place_market_trade(
    symbol: str = Body(..., embed=True),
    action: str = Body(..., embed=True),
    volume: float = Body(0.1, embed=True),
    sl: float = Body(0.0, embed=True),
    tp: float = Body(0.0, embed=True),
    sl_points: int = Body(None, embed=True),
    tp_points: int = Body(None, embed=True),
    magic: int = Body(AGENT_MAGIC, embed=True),
):
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        return {
            "success": False,
            "error": (
                f"Symbol {symbol!r} not found in MT5. In the terminal: View → Market Watch → "
                "right‑click → Symbols, search BTC, and set your app to that exact name."
            ),
        }

    symbol_upper = resolved
    action_upper = action.upper()
    
    existing_positions = mt5.positions_get(symbol=symbol_upper)
    if existing_positions:
        agent_positions = [p for p in existing_positions if int(p.magic) == int(magic)]
        if agent_positions:
            return {
                "success": False,
                "error": f"An open {symbol_upper} position already exists. Close it before opening another.",
                "open_tickets": [int(p.ticket) for p in agent_positions],
            }

    symbol_info = mt5.symbol_info(symbol_upper)
    if symbol_info is None:
        return {"success": False, "error": f"Symbol {symbol_upper} not found in MT5."}
        
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol_upper, True):
            return {"success": False, "error": f"Failed to select/show symbol {symbol_upper}."}

    tick = mt5.symbol_info_tick(symbol_upper)
    if tick is None:
        return {"success": False, "error": f"No live tick available for {symbol_upper}."}

    if action_upper == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif action_upper == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        return {"success": False, "error": "Invalid action parameter. Must be 'BUY' or 'SELL'."}

    sl_price, tp_price = _resolve_sl_tp_prices(
        action_upper, price, symbol_info.point, sl, tp, sl_points, tp_points
    )

    if sl_price > 0:
        if action_upper == "BUY" and sl_price >= price:
            return {"success": False, "error": "BUY stop loss must be below entry price."}
        if action_upper == "SELL" and sl_price <= price:
            return {"success": False, "error": "SELL stop loss must be above entry price."}
    if tp_price > 0:
        if action_upper == "BUY" and tp_price <= price:
            return {"success": False, "error": "BUY take profit must be above entry price."}
        if action_upper == "SELL" and tp_price >= price:
            return {"success": False, "error": "SELL take profit must be below entry price."}

    # Enforce broker minimum stop distance when stops are provided
    min_stop_dist = symbol_info.trade_stops_level * symbol_info.point
    if min_stop_dist > 0:
        if sl_price > 0 and abs(price - sl_price) < min_stop_dist:
            return {
                "success": False,
                "error": f"Stop loss too close to market (min {min_stop_dist} price units).",
            }
        if tp_price > 0 and abs(price - tp_price) < min_stop_dist:
            return {
                "success": False,
                "error": f"Take profit too close to market (min {min_stop_dist} price units).",
            }

    order_volume = normalize_volume(float(volume), symbol_info)

    filling_type = mt5.ORDER_FILLING_FOK
    if symbol_info.filling_mode & SYMBOL_FILLING_FOK:
        filling_type = mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & SYMBOL_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        # Fallback for accounts that require standard execution returns (e.g. many CFD/Crypto brokers)
        filling_type = mt5.ORDER_FILLING_RETURN

    trade_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_upper,
        "volume": order_volume,
        "type": order_type,
        "price": float(price),
        "sl": round(float(sl_price), symbol_info.digits) if sl_price > 0 else 0.0,
        "tp": round(float(tp_price), symbol_info.digits) if tp_price > 0 else 0.0,
        "deviation": 20,
        "magic": int(magic),
        "comment": "Sent via FastAPI Gateway",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    result = mt5.order_send(trade_request)

    if result is None:
        return {"success": False, "error": "Order send call timed out or failed to communicate with terminal."}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {
            "success": False,
            "error": f"Trade execution rejected by broker server.",
            "mt5_retcode": result.retcode,
            "comment": result.comment
        }

    return {
        "success": True,
        "message": f"Successfully executed live market {action_upper} order.",
        "order_id": result.order,
        "volume": order_volume,
        "execution_price": result.price,
        "sl": round(float(sl_price), symbol_info.digits) if sl_price > 0 else 0.0,
        "tp": round(float(tp_price), symbol_info.digits) if tp_price > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SL GUARDIAN
#
# Two-mode tick loop (runs every 0.5 s):
#
#   PROTECTION  — when price moves `danger_fraction` of the way from entry
#                 toward SL the position is closed at market immediately,
#                 saving part of the originally risked capital.
#
#   STACKING    — while a position is open and has already moved at least
#                 `stack_profit_threshold` of the way toward TP, the guardian
#                 scans for a fresh same-direction reversal signal.  If one is
#                 found (and the total open positions on that symbol is below
#                 `max_stack_positions`) a `stack_signal` event is emitted so
#                 the frontend can open an additional position and ride the move
#                 harder.
# ─────────────────────────────────────────────────────────────────────────────

class SlGuardianConfig(BaseModel):
    enabled: bool = True
    # Close early when adverse move equals this fraction of total SL distance
    danger_fraction: float = Field(0.70, ge=0.05, le=0.99)
    # Ignore positions younger than this many seconds (avoids spread noise)
    min_age_secs: int = Field(0, ge=0)
    # Maximum number of stacked positions allowed per symbol (1 = stacking off)
    max_stack_positions: int = Field(2, ge=1, le=5)
    # Position must have moved this fraction toward TP before stacking is allowed
    stack_profit_threshold: float = Field(0.30, ge=0.05, le=0.90)
    magic: int = Field(AGENT_MAGIC)


class WatchPositionIn(BaseModel):
    ticket: int = Field(..., gt=0)
    symbol: str
    side: str                           # "BUY" | "SELL"
    entry: float = Field(..., gt=0)
    sl: float = Field(..., gt=0)
    tp: float = Field(0.0)
    liquidity_price: Optional[float] = None   # original sweep level if known
    level_type: Optional[str] = None          # "BSL" | "SSL"
    magic: int = AGENT_MAGIC


class _PositionMeta(TypedDict):
    ticket: int
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    magic: int
    liquidity_price: Optional[float]
    level_type: Optional[str]
    opened_at: float


class SlGuardian:
    """
    Tick-level SL proximity monitor with position-stacking engine.

    Usage:
      1. After placing a trade call ``sl_guardian.watch(meta)`` with the
         ticket, symbol, side, entry, sl, tp and optional liquidity info.
      2. The internal loop checks price every 0.5 s.
         - PROTECTION: when price moves >= danger_fraction × SL distance
           the position is closed at market immediately.
         - STACKING: when the position has moved >= stack_profit_threshold
           toward TP and a fresh same-direction reversal signal is detected,
           a ``stack_signal`` event fires so the frontend can add to the
           winner.
    """

    def __init__(self) -> None:
        self._config = SlGuardianConfig()
        self._watched: dict[int, _PositionMeta] = {}
        # De-duplication set for stack signals — keyed by ticket+signal identity
        self._stack_emitted: set[str] = set()
        self._log: list[dict] = []
        self._task: Optional[asyncio.Task] = None
        self._subscribers: list[asyncio.Queue] = []

    # ── public helpers ────────────────────────────────────────────────────────

    def configure(self, cfg: SlGuardianConfig) -> None:
        self._config = cfg

    def get_config(self) -> dict:
        return self._config.model_dump()

    def watch(self, meta: _PositionMeta) -> None:
        self._watched[meta["ticket"]] = meta
        self._emit({"type": "watch_added", "ticket": meta["ticket"], "symbol": meta["symbol"]})

    def unwatch(self, ticket: int) -> None:
        removed = self._watched.pop(ticket, None)
        if removed:
            self._emit({"type": "watch_removed", "ticket": ticket})

    def get_status(self) -> dict:
        return {
            "enabled": self._config.enabled,
            "danger_fraction": self._config.danger_fraction,
            "min_age_secs": self._config.min_age_secs,
            "max_stack_positions": self._config.max_stack_positions,
            "stack_profit_threshold": self._config.stack_profit_threshold,
            "watched_count": len(self._watched),
            "watched": list(self._watched.values()),
            "recent_events": self._log[-30:],
        }

    # ── pub/sub ───────────────────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers = [x for x in self._subscribers if x is not q]

    def _emit(self, event: dict) -> None:
        stamped = {**event, "_ts": datetime.now(timezone.utc).isoformat()}
        self._log.append(stamped)
        if len(self._log) > 500:
            self._log = self._log[-500:]
        for q in self._subscribers:
            try:
                q.put_nowait(stamped)
            except asyncio.QueueFull:
                pass

    # ── danger check ──────────────────────────────────────────────────────────

    def _in_danger(self, meta: _PositionMeta, bid: float, ask: float) -> bool:
        """True when price has moved ≥ danger_fraction × risk toward SL."""
        entry = meta["entry"]
        sl = meta["sl"]
        if sl <= 0:
            return False
        total_risk = abs(entry - sl)
        if total_risk < 1e-12:
            return False
        # For a BUY the adverse price is the current bid; for SELL it is ask.
        current = bid if meta["side"] == "BUY" else ask
        adverse = (entry - current) if meta["side"] == "BUY" else (current - entry)
        if adverse <= 0:
            return False
        return (adverse / total_risk) >= self._config.danger_fraction

    # ── MT5 close helper (called synchronously from async loop) ──────────────

    def _close_position_now(self, ticket: int) -> Tuple[Optional[float], str]:
        """Market-close a position. Returns (fill_price, error_msg)."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return None, "Position not found"
        pos = positions[0]
        symbol = pos.symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None, "symbol_info unavailable"
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None, "No tick"

        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask

        filling_type = mt5.ORDER_FILLING_FOK
        if symbol_info.filling_mode & SYMBOL_FILLING_FOK:
            filling_type = mt5.ORDER_FILLING_FOK
        elif symbol_info.filling_mode & SYMBOL_FILLING_IOC:
            filling_type = mt5.ORDER_FILLING_IOC
        else:
            filling_type = mt5.ORDER_FILLING_RETURN

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos.volume),
            "type": close_type,
            "position": int(ticket),
            "price": float(price),
            "deviation": 20,
            "magic": int(pos.magic),
            "comment": "SL Guardian early exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return float(result.price), ""
        err = f"retcode={result.retcode if result else 'None'} {mt5.last_error()}"
        return None, err

    # ── monitoring loop ───────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(0.5)
                if not self._config.enabled:
                    continue
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                self._emit({"type": "loop_error", "error": str(exc)})

    async def _tick(self) -> None:
        now_ts = datetime.now(timezone.utc).timestamp()

        # ── 1. protection — monitor watched positions for early exit ──────────
        all_positions = mt5.positions_get()
        open_tickets = {int(p.ticket) for p in (all_positions or [])}

        for ticket in list(self._watched):
            meta = self._watched[ticket]

            # Already closed externally (broker hit SL/TP)
            if ticket not in open_tickets:
                self._emit({"type": "position_closed_externally", "ticket": ticket, "symbol": meta["symbol"]})
                del self._watched[ticket]
                continue

            # Respect minimum age
            if now_ts - meta["opened_at"] < self._config.min_age_secs:
                continue

            tick = mt5.symbol_info_tick(meta["symbol"])
            if tick is None:
                continue

            if not self._in_danger(meta, float(tick.bid), float(tick.ask)):
                continue

            # ── danger zone hit → close early ─────────────────────────────────
            fill_price, err = self._close_position_now(ticket)
            if fill_price is not None:
                saved = abs(fill_price - meta["sl"])
                self._emit({
                    "type": "early_exit",
                    "ticket": ticket,
                    "symbol": meta["symbol"],
                    "side": meta["side"],
                    "entry": meta["entry"],
                    "sl": meta["sl"],
                    "exit_price": fill_price,
                    "saved_distance": round(saved, 8),
                    "danger_fraction_reached": self._config.danger_fraction,
                })
                del self._watched[ticket]
                # Remove any pending stack keys for this ticket so a future
                # fresh position isn't accidentally deduplicated.
                self._stack_emitted = {k for k in self._stack_emitted if not k.startswith(f"{ticket}-")}
            else:
                self._emit({"type": "early_exit_failed", "ticket": ticket, "symbol": meta["symbol"], "error": err})

        # ── 2. stacking — add to winners when a new confirming signal fires ───
        if self._config.max_stack_positions <= 1 or not self._watched:
            return

        # Bound the de-dup set to avoid unbounded growth
        if len(self._stack_emitted) > 1000:
            self._stack_emitted = set(list(self._stack_emitted)[-500:])

        for ticket, meta in list(self._watched.items()):
            symbol = meta["symbol"]
            resolved = resolve_mt5_symbol(symbol)
            if resolved is None:
                continue

            # Count total agent positions already open on this symbol
            existing = mt5.positions_get(symbol=resolved)
            open_count = len([
                p for p in (existing or [])
                if int(p.magic) in {AGENT_MAGIC, M1_PUSH_SCALP_MAGIC}
            ])
            if open_count >= self._config.max_stack_positions:
                continue

            # Position must have moved at least stack_profit_threshold toward TP
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            current = float(tick.bid) if meta["side"] == "BUY" else float(tick.ask)
            entry = meta["entry"]
            tp = meta["tp"]
            if tp <= 0:
                continue
            tp_distance = abs(tp - entry)
            if tp_distance < 1e-12:
                continue
            favorable_move = (current - entry) if meta["side"] == "BUY" else (entry - current)
            profit_fraction = favorable_move / tp_distance
            if profit_fraction < self._config.stack_profit_threshold:
                continue

            # Scan for a fresh same-direction reversal signal (last 5 minutes only)
            candles, cerr = copy_rates_as_candles(resolved, "M1", 60)
            if cerr or not candles:
                continue
            df = pd.DataFrame(candles)
            liquidity = calculate_advanced_liquidity(df, window=5, source_tag="Local")
            confirm_candles, _ = copy_rates_as_candles(resolved, "M5", 20)
            rev_result = analyze_live_reversals(
                candles,
                confirm_candles if confirm_candles else candles,
                liquidity,
                chart_timeframe="M1",
                trend_filter=False,
            )

            all_signals: list[dict] = rev_result.get("signals", []) + rev_result.get("predictions", [])
            fresh = [
                s for s in all_signals
                if s["side"] == meta["side"]
                and s.get("reversalTime", 0) > now_ts - 300
            ]
            if not fresh:
                continue

            sig = fresh[0]
            stack_key = f"{ticket}-{sig.get('reversalTime', 0)}-{sig.get('liquidityPrice', 0)}"
            if stack_key in self._stack_emitted:
                continue
            self._stack_emitted.add(stack_key)

            self._emit({
                "type": "stack_signal",
                "anchor_ticket": ticket,
                "symbol": symbol,
                "side": meta["side"],
                "anchor_entry": entry,
                "anchor_sl": meta["sl"],
                "anchor_tp": tp,
                "current_price": round(current, 8),
                "profit_fraction": round(profit_fraction, 3),
                "open_count": open_count,
                "max_stack": self._config.max_stack_positions,
                "signal": sig,
            })

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None


sl_guardian = SlGuardian()


@app.on_event("startup")
async def _start_sl_guardian() -> None:
    sl_guardian.start()


# ── SL Guardian REST endpoints ────────────────────────────────────────────────

@app.post("/api/sl-guardian/configure")
async def sl_guardian_configure(cfg: SlGuardianConfig):
    """Update SL Guardian thresholds at runtime."""
    sl_guardian.configure(cfg)
    return {"success": True, "config": sl_guardian.get_config()}


@app.get("/api/sl-guardian/status")
async def sl_guardian_status():
    """Return watched positions, active ghosts, and recent event log."""
    return sl_guardian.get_status()


@app.post("/api/sl-guardian/watch")
async def sl_guardian_watch(req: WatchPositionIn):
    """
    Register a live position with the SL Guardian.
    Call this immediately after a trade is placed so the guardian starts
    monitoring it for early exit.
    """
    resolved = resolve_mt5_symbol(req.symbol)
    if resolved is None:
        return {"success": False, "error": f"Symbol {req.symbol!r} not found in MT5."}
    side = req.side.upper()
    if side not in ("BUY", "SELL"):
        return {"success": False, "error": "side must be BUY or SELL"}

    meta: _PositionMeta = {
        "ticket": req.ticket,
        "symbol": resolved,
        "side": side,
        "entry": float(req.entry),
        "sl": float(req.sl),
        "tp": float(req.tp),
        "magic": int(req.magic),
        "liquidity_price": float(req.liquidity_price) if req.liquidity_price is not None else None,
        "level_type": req.level_type,
        "opened_at": datetime.now(timezone.utc).timestamp(),
    }
    sl_guardian.watch(meta)
    return {"success": True, "ticket": req.ticket, "symbol": resolved}


@app.delete("/api/sl-guardian/watch/{ticket}")
async def sl_guardian_unwatch(ticket: int):
    """Remove a position from the SL Guardian (e.g. after manual TP close)."""
    sl_guardian.unwatch(ticket)
    return {"success": True, "ticket": ticket}


@app.websocket("/ws/sl-guardian")
async def ws_sl_guardian(websocket: WebSocket):
    """
    Stream SL Guardian events in real-time.

    Event types:
      watch_added                — position registered
      watch_removed              — position manually un-watched
      position_closed_externally — broker hit SL or TP before guardian could act
      early_exit                 — guardian closed the position early; includes saved_distance
      early_exit_failed          — MT5 rejected the close order
      stack_signal               — position profitable enough to stack; new same-direction
                                   reversal signal detected; includes anchor_ticket + signal
      loop_error                 — internal guardian error (rare)
    """
    await websocket.accept()
    q = sl_guardian.subscribe()
    try:
        # Send current status snapshot on connect
        await websocket.send_json({"type": "snapshot", **sl_guardian.get_status()})
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Heartbeat so the connection stays alive
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        sl_guardian.unsubscribe(q)


@app.post("/api/close")
async def close_agent_position(
    ticket: int = Body(..., embed=True),
    magic: int = Body(AGENT_MAGIC, embed=True),
):
    """Close a single agent-managed position at market (used for liquidity flip)."""
    positions = mt5.positions_get(ticket=int(ticket))
    if not positions:
        return {"success": False, "error": f"Position {ticket} not found."}

    pos = positions[0]
    allowed = {AGENT_MAGIC, M1_PUSH_SCALP_MAGIC}
    if int(pos.magic) not in allowed or int(pos.magic) != int(magic):
        return {"success": False, "error": "Position magic does not match this agent."}

    symbol_upper = pos.symbol
    symbol_info = mt5.symbol_info(symbol_upper)
    if symbol_info is None:
        return {"success": False, "error": f"Symbol {symbol_upper} not found."}

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol_upper, True):
            return {"success": False, "error": f"Failed to select symbol {symbol_upper}."}

    tick = mt5.symbol_info_tick(symbol_upper)
    if tick is None:
        return {"success": False, "error": f"No live tick for {symbol_upper}."}

    if pos.type == mt5.POSITION_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    filling_type = mt5.ORDER_FILLING_FOK
    if symbol_info.filling_mode & SYMBOL_FILLING_FOK:
        filling_type = mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & SYMBOL_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        filling_type = mt5.ORDER_FILLING_RETURN

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_upper,
        "volume": float(pos.volume),
        "type": close_type,
        "position": int(ticket),
        "price": float(price),
        "deviation": 20,
        "magic": AGENT_MAGIC,
        "comment": "Agent liquidity flip close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    result = mt5.order_send(close_request)
    if result is None:
        return {"success": False, "error": "Close order failed to reach terminal."}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {
            "success": False,
            "error": "Broker rejected position close.",
            "mt5_retcode": result.retcode,
            "comment": result.comment,
        }

    return {
        "success": True,
        "message": f"Closed position {ticket}.",
        "execution_price": float(result.price),
        "volume": float(pos.volume),
    }