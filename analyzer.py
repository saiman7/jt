import MetaTrader5 as mt5
import time
from collections import deque

# initialize MT5
mt5.initialize()

symbol = "XAUUSD"
mt5.symbol_select(symbol, True)

# store last 30 ticks
ticks = deque(maxlen=30)

print("Sweep + Reversal Engine Started...")

# thresholds (you will tune later)
SWEEP_THRESHOLD = 5      # fast move size
IMPULSE_THRESHOLD = 7    # reversal strength
STALL_LIMIT = 3          # small movement zone

state = "IDLE"
sweep_direction = None
sweep_price = None
stall_counter = 0

while True:
    tick = mt5.symbol_info_tick(symbol)

    if tick:
        price = tick.bid
        ticks.append(price)

        if len(ticks) < 10:
            time.sleep(0.05)
            continue

        # window movement
        move = ticks[-1] - ticks[0]

        # -------------------------
        # 1. DETECT SWEEP
        # -------------------------
        if state == "IDLE":
            if move <= -SWEEP_THRESHOLD:
                state = "SWEEP_DOWN"
                sweep_direction = "DOWN"
                sweep_price = price
                stall_counter = 0
                print("🧲 Liquidity Sweep DOWN detected")

            elif move >= SWEEP_THRESHOLD:
                state = "SWEEP_UP"
                sweep_direction = "UP"
                sweep_price = price
                stall_counter = 0
                print("🧲 Liquidity Sweep UP detected")

        # -------------------------
        # 2. STALL PHASE
        # -------------------------
        elif state in ["SWEEP_DOWN", "SWEEP_UP"]:
            recent_move = ticks[-1] - ticks[-5]

            if abs(recent_move) < 1:
                stall_counter += 1
            else:
                stall_counter = 0

            # -------------------------
            # 3. REVERSAL DETECTION
            # -------------------------
            if state == "SWEEP_DOWN":
                reversal_move = ticks[-1] - sweep_price

                if reversal_move >= IMPULSE_THRESHOLD and stall_counter >= STALL_LIMIT:
                    print("🔥 BUY SIGNAL (Reversal after DOWN sweep)")
                    state = "IDLE"

            elif state == "SWEEP_UP":
                reversal_move = sweep_price - ticks[-1]

                if reversal_move >= IMPULSE_THRESHOLD and stall_counter >= STALL_LIMIT:
                    print("🔥 SELL SIGNAL (Reversal after UP sweep)")
                    state = "IDLE"

        # reset if no structure
        if len(ticks) == 30:
            ticks.popleft()

    time.sleep(0.05)