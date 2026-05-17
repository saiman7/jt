import asyncio
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware

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

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def calculate_advanced_liquidity(df, window=5, source_tag="Local"):
    liquidity_segments = []
    total_candles = len(df)
    seen_prices = set()

    for i in range(window, total_candles - window):
        current_high = float(df.iloc[i]['high'])
        current_low = float(df.iloc[i]['low'])
        start_ts = int(df.iloc[i]['time'])

        # 1. Swing High (BSL)
        if current_high > df.iloc[i-window:i]['high'].max() and current_high >= df.iloc[i+1:i+window+1]['high'].max():
            rounded_price = round(current_high, 2)
            if rounded_price not in seen_prices:
                seen_prices.add(rounded_price)
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
            if rounded_price not in seen_prices:
                seen_prices.add(rounded_price)
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
# UPGRADED ROUTE: MULTI-TIMEFRAME LIQUIDITY AGGREGATOR
# -------------------------------------------------------------
@app.get("/api/historical")
async def get_historical_backtest_data(
    symbol: str = Query("BTCUSD"), 
    timeframe: str = Query("M5"), 
    start_time: str = Query(None),  # Expects: ISO String (YYYY-MM-DDTHH:MM) or Unix Timestamp
    end_time: str = Query(None)
):
    symbol = symbol.upper()
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

@app.websocket("/ws/rates")
async def websocket_rates_endpoint(websocket: WebSocket, symbol: str = "BTCUSD", timeframe: str = "M1"):
    await websocket.accept()
    symbol = symbol.upper()
    mt5_tf = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M1)
    
    try:
        while True:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 250)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                advanced_liquidity = calculate_advanced_liquidity(df, window=5, source_tag="Local")
                
                payload = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": df[["time", "open", "high", "low", "close"]].to_dict(orient="records"),
                    "liquidity": advanced_liquidity[-10:] 
                }
                await websocket.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

@app.post("/api/trade")
async def place_market_trade(
    symbol: str = Body(..., embed=True),
    action: str = Body(..., embed=True),
    volume: float = Body(0.1, embed=True),
    sl_points: int = Body(None, embed=True),
    tp_points: int = Body(None, embed=True)
):
    symbol_upper = symbol.upper()
    action_upper = action.upper()
    
    symbol_info = mt5.symbol_info(symbol_upper)
    if symbol_info is None:
        return {"success": False, "error": f"Symbol {symbol_upper} not found in MT5."}
        
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol_upper, True):
            return {"success": False, "error": f"Failed to select/show symbol {symbol_upper}."}

    if action_upper == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol_upper).ask
        sl_price = price - (sl_points * symbol_info.point) if sl_points else 0.0
        tp_price = price + (tp_points * symbol_info.point) if tp_points else 0.0
    elif action_upper == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol_upper).bid
        sl_price = price + (sl_points * symbol_info.point) if sl_points else 0.0
        tp_price = price - (tp_points * symbol_info.point) if tp_points else 0.0
    else:
        return {"success": False, "error": "Invalid action parameter. Must be 'BUY' or 'SELL'."}

    filling_type = mt5.ORDER_FILLING_FOK
    if symbol_info.filling_mode == mt5.SYMBOL_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    elif symbol_info.filling_mode == mt5.SYMBOL_FILLING_BOC:
        filling_type = mt5.ORDER_FILLING_BOC

    trade_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_upper,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": round(float(sl_price), symbol_info.digits) if sl_price > 0 else 0.0,
        "tp": round(float(tp_price), symbol_info.digits) if tp_price > 0 else 0.0,
        "deviation": 20,
        "magic": 123456,
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
        "volume": result.volume,
        "execution_price": result.price
    }