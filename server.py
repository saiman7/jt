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
    "D1": mt5.TIMEFRAME_D1,
}

def calculate_advanced_liquidity(df, window=5):
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
                    "type": "BSL", "price": current_high, "start_time": start_ts, "end_time": end_ts, "is_grabbed": has_been_swept
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
                    "type": "SSL", "price": current_low, "start_time": start_ts, "end_time": end_ts, "is_grabbed": has_been_swept
                })

    return liquidity_segments

# -------------------------------------------------------------
# FIXED ROUTE: TARGETED SIMULATION RANGE COPIER
# -------------------------------------------------------------
@app.get("/api/historical")
async def get_historical_backtest_data(
    symbol: str = Query("BTCUSD"), 
    timeframe: str = Query("M5"), 
    start_time: str = Query(None),  # 💡 Expects: ISO String (YYYY-MM-DDTHH:MM) or Unix Timestamp
    end_time: str = Query(None)
):
    symbol = symbol.upper()
    mt5_tf = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M5)
    
    # Process custom date ranges if they are supplied by the frontend interface
    if start_time and end_time:
        try:
            # Handle ISO string transformations safely
            if "T" in start_time:
                date_start = datetime.fromisoformat(start_time)
                date_end = datetime.fromisoformat(end_time)
            else:
                date_start = datetime.fromtimestamp(int(start_time))
                date_end = datetime.fromtimestamp(int(end_time))
                
            # 💡 CRITICAL: Fetch 50 extra baseline candles BEFORE your start date 
            # This handles lookback padding so indicators/liquidity process instantly at frame 1
            minutes_padding = 50 * int(timeframe.replace("M", "").replace("H", "60").replace("D", "1440"))
            padded_start = datetime.fromtimestamp(int(date_start.timestamp()) - (minutes_padding * 60))

            rates = mt5.copy_rates_range(symbol, mt5_tf, padded_start, date_end)
        except Exception as err:
            return {"error": f"Date parsing layout breakdown: {str(err)}"}
    else:
        # Fallback to general default matrix behavior if fields are left blank
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 1000)
    
    if rates is None or len(rates) == 0:
        return {"error": f"No historical records found inside specified parameters for symbol: {symbol}"}
        
    df = pd.DataFrame(rates)
    all_liquidity_levels = calculate_advanced_liquidity(df, window=5)
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": df[["time", "open", "high", "low", "close"]].to_dict(orient="records"),
        "liquidity": all_liquidity_levels
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
                advanced_liquidity = calculate_advanced_liquidity(df, window=5)
                
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
    action: str = Body(..., embed=True),  # Expects: "BUY" or "SELL"
    volume: float = Body(0.1, embed=True), # Lot size
    sl_points: int = Body(None, embed=True), # Optional Stop Loss in Points (e.g., 200)
    tp_points: int = Body(None, embed=True)  # Optional Take Profit in Points (e.g., 400)
):
    symbol_upper = symbol.upper()
    action_upper = action.upper()
    
    # 1. Verify symbol availability in MT5 terminal
    symbol_info = mt5.symbol_info(symbol_upper)
    if symbol_info is None:
        return {"success": False, "error": f"Symbol {symbol_upper} not found in MT5."}
        
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol_upper, True):
            return {"success": False, "error": f"Failed to select/show symbol {symbol_upper}."}

    # 2. Determine trade direction parameters and live execution prices
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

    # 3. Detect broker-supported filling type execution policy dynamically
    # (Solves common 'Unsupported Filling Mode' error 10030)
    filling_type = mt5.ORDER_FILLING_FOK
    if symbol_info.filling_mode == mt5.SYMBOL_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    elif symbol_info.filling_mode == mt5.SYMBOL_FILLING_BOC:
        filling_type = mt5.ORDER_FILLING_BOC

    # 4. Construct the trade request payload structural matrix
    trade_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_upper,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": round(float(sl_price), symbol_info.digits) if sl_price > 0 else 0.0,
        "tp": round(float(tp_price), symbol_info.digits) if tp_price > 0 else 0.0,
        "deviation": 20,
        "magic": 123456,  # Identifier tag for trades placed by this API
        "comment": "Sent via FastAPI Gateway",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    # 5. Fire trade execution command straight into the live terminal deal desk
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