import MetaTrader5 as mt5
import csv
from collections import deque
from datetime import datetime, timedelta

# ----------------------------
# INIT
# ----------------------------
mt5.initialize()

symbol = "XAUUSD"
mt5.symbol_select(symbol, True)

# ----------------------------
# LOAD 1 DAY TICKS (PAST DATA)
# ----------------------------
end_time = datetime.now()
start_time = end_time - timedelta(days=1)

ticks_data = mt5.copy_ticks_range(
    symbol,
    start_time,
    end_time,
    mt5.COPY_TICKS_ALL
)

if ticks_data is None or len(ticks_data) == 0:
    print("No tick data found")
    quit()

print(f"Loaded ticks: {len(ticks_data)}")

# ----------------------------
# CSV OUTPUT
# ----------------------------
file = open("backtest_trades.csv", mode="w", newline="")
writer = csv.writer(file)

writer.writerow([
    "entry_time",
    "exit_time",
    "direction",
    "entry_price",
    "exit_price",
    "pnl_ticks",
    "reason"
])

# ----------------------------
# STRATEGY SETTINGS
# ----------------------------
ticks = deque(maxlen=50)

SWEEP_THRESHOLD = 5
IMPULSE_THRESHOLD = 7
STALL_LIMIT = 3

TP_TICKS = 10
SL_TICKS = 6

# ----------------------------
# STATE
# ----------------------------
state = "IDLE"
sweep_price = None
stall_counter = 0

position = None

# ----------------------------
# BACKTEST LOOP (REPLAY TICKS)
# ----------------------------
for tick in ticks_data:

    price = tick['bid']
    t = tick['time']

    ticks.append(price)

    if len(ticks) < 10:
        continue

    move = ticks[-1] - ticks[0]

    # ----------------------------
    # ENTRY LOGIC
    # ----------------------------
    if position is None:

        if state == "IDLE":

            if move <= -SWEEP_THRESHOLD:
                state = "SWEEP_DOWN"
                sweep_price = price
                stall_counter = 0

            elif move >= SWEEP_THRESHOLD:
                state = "SWEEP_UP"
                sweep_price = price
                stall_counter = 0

        elif state in ["SWEEP_DOWN", "SWEEP_UP"]:

            recent_move = ticks[-1] - ticks[-5]

            if abs(recent_move) < 1:
                stall_counter += 1
            else:
                stall_counter = 0

            # BUY
            if state == "SWEEP_DOWN":
                reversal_move = ticks[-1] - sweep_price

                if reversal_move >= IMPULSE_THRESHOLD and stall_counter >= STALL_LIMIT:
                    position = {
                        "direction": "BUY",
                        "entry_price": price,
                        "entry_time": t
                    }
                    state = "IDLE"

            # SELL
            elif state == "SWEEP_UP":
                reversal_move = sweep_price - ticks[-1]

                if reversal_move >= IMPULSE_THRESHOLD and stall_counter >= STALL_LIMIT:
                    position = {
                        "direction": "SELL",
                        "entry_price": price,
                        "entry_time": t
                    }
                    state = "IDLE"

    # ----------------------------
    # POSITION MANAGEMENT
    # ----------------------------
    else:

        entry = position["entry_price"]
        direction = position["direction"]

        if direction == "BUY":
            pnl = price - entry
        else:
            pnl = entry - price

        # TAKE PROFIT
        if pnl >= TP_TICKS:
            writer.writerow([
                position["entry_time"],
                t,
                direction,
                entry,
                price,
                pnl,
                "TAKE_PROFIT"
            ])

            position = None

        # STOP LOSS
        elif pnl <= -SL_TICKS:
            writer.writerow([
                position["entry_time"],
                t,
                direction,
                entry,
                price,
                pnl,
                "STOP_LOSS"
            ])

            position = None

print("Backtest completed → backtest_trades.csv")