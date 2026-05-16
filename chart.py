import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
import time

mt5.initialize()

st.set_page_config(layout="wide")

SYMBOL = "BTCUSD"

timeframe_option = st.selectbox(
    "Timeframe",
    ["M1", "M5", "M15", "H1"]
)

tf_map = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

TIMEFRAME = tf_map[timeframe_option]

# 🔥 ALWAYS FRESH FETCH (NO CACHE)
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1000)

df = pd.DataFrame(rates)

if df.empty:
    st.error("No data from MT5. Check symbol or connection.")
    st.stop()

df["time"] = pd.to_datetime(df["time"], unit="s")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
))

fig.update_layout(
    height=800,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    yaxis=dict(side="right")
)

st.plotly_chart(fig, use_container_width=True)

st.write(f"""
Timeframe: {timeframe_option}  
Candles: {len(df)}
""")

time.sleep(1)
st.rerun()