import asyncio
import math
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Ensure `jt/` is on sys.path when uvicorn is started outside this folder
_JT_DIR = Path(__file__).resolve().parent
if str(_JT_DIR) not in sys.path:
    sys.path.insert(0, str(_JT_DIR))
from typing import Any, List, Literal, Optional, Tuple
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
from a_plus_context import (
    compute_a_plus_context,
    profile_for_symbol,
    validate_a_plus_entry,
)

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


# -------------------------------------------------------------
# A+ context (HTF bias + MTF zone) — drives strict entry gate
# -------------------------------------------------------------
def build_a_plus_context_for_symbol(symbol: str, fallback_price: Optional[float] = None) -> dict:
    """
    Fetch HTF + MTF bars per the symbol profile and return the A+ context.
    Returns {"error": ...} on failure so the frontend can degrade gracefully.
    """
    profile = profile_for_symbol(symbol)
    htf_candles, htf_err = copy_rates_as_candles(symbol, profile["htf"], profile["htf_count"])
    mtf_candles, mtf_err = copy_rates_as_candles(symbol, profile["mtf"], profile["mtf_count"])

    if htf_err or mtf_err:
        return {
            "error": htf_err or mtf_err,
            "htfTimeframe": profile["htf"],
            "mtfTimeframe": profile["mtf"],
            "htf": {"bias": "NEUTRAL", "reason": "HTF bars unavailable"},
            "mtf": {"zone": "UNKNOWN", "reason": "MTF bars unavailable"},
        }

    tick = mt5.symbol_info_tick(symbol)
    if tick is not None:
        current_price = float((tick.bid + tick.ask) / 2.0)
    elif fallback_price is not None:
        current_price = float(fallback_price)
    elif mtf_candles:
        current_price = float(mtf_candles[-1]["close"])
    else:
        current_price = 0.0

    ctx = compute_a_plus_context(htf_candles, mtf_candles, current_price, profile=profile)
    ctx["currentPrice"] = current_price
    return ctx


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
    combined = calculate_advanced_liquidity(df_chart, window=5, source_tag="Local")

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


@app.get("/api/a-plus-context")
async def a_plus_context_endpoint(symbol: str = Query(...)):
    """
    Current A+ context (HTF bias + MTF premium/discount/equilibrium) for a symbol.
    Used by the live agent UI to display gate status and by debugging.
    """
    resolved = resolve_mt5_symbol(symbol)
    if resolved is None:
        return {"error": f"No MT5 symbol matching {symbol!r}."}
    ctx = build_a_plus_context_for_symbol(resolved)
    ctx["symbol"] = resolved
    return ctx


class APlusValidateBody(BaseModel):
    side: Literal["BUY", "SELL"]
    symbol: str


@app.post("/api/a-plus-validate")
async def a_plus_validate_endpoint(body: APlusValidateBody):
    """Return whether a hypothetical entry passes the A+ gate right now."""
    resolved = resolve_mt5_symbol(body.symbol)
    if resolved is None:
        return {"error": f"No MT5 symbol matching {body.symbol!r}."}
    ctx = build_a_plus_context_for_symbol(resolved)
    return {"symbol": resolved, "context": ctx, "verdict": validate_a_plus_entry(body.side, ctx)}


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
                combined_liquidity = advanced_liquidity[-10:]

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
                        trend_filter=False,
                    )
                    payload["formingConfirmCandle"] = reversal.get("formingConfirmCandle")
                    payload["reversalSignals"] = reversal.get("signals", [])
                    payload["reversalPredictions"] = reversal.get("predictions", [])
                    payload["confirmTimeframe"] = reversal.get("confirmTimeframe")

                # A+ context (HTF bias + MTF zone) attached on every tick so the
                # client can strictly gate entries against the doc rules.
                fallback_price = float(chart_candles[-1]["close"]) if chart_candles else None
                payload["aPlusContext"] = build_a_plus_context_for_symbol(symbol, fallback_price)

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