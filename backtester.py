import MetaTrader5 as mt5
import csv
from datetime import datetime, timedelta, timezone
import itertools
from typing import Optional, Tuple

mt5.initialize()

symbol = "XAUUSD"
if not mt5.symbol_select(symbol, True):
    print(f"Could not select {symbol}")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# DEMO / LIVE TRUTH: volume, contract, spread, commission → $ P&L
# ---------------------------------------------------------------------------

# "fixed"  → always trade FIXED_LOT
# "risk"   → risk RISK_PERCENT of balance per trade; lot size from SL distance
SIZING_MODE = "risk"  # or "fixed"

# Used when SIZING_MODE == "fixed"
FIXED_LOT = 0.1

# Used when SIZING_MODE == "risk" (see SIMULATE_ACCOUNT_BALANCE below)
RISK_PERCENT = 3.0  # % of balance at risk if full SL is hit
ACCOUNT_BALANCE_FALLBACK = 100_000.0  # if account_info unavailable

# None = use live MT5 balance. Set e.g. 10.0 to stress-test a tiny account.
SIMULATE_ACCOUNT_BALANCE = 400.0

# Rough margin check: notional / leverage when MT5 returns nonsense leverage
ASSUMED_LEVERAGE = 100
MARGIN_BUFFER = 1.02  # equity must exceed margin × this to open

# Extra realism (per 1.0 lot, full round-trip in account currency; 0 if unknown)
COMMISSION_PER_LOT_ROUNDTRIP = 0.0

# Subtract approximate spread cost each close (uses live symbol spread from MT5)
DEDUCT_SPREAD_EACH_TRADE = True

# ---------------------------------------------------------------------------
# STRATEGY PARAMS
# ---------------------------------------------------------------------------
FIXED_PARAMS = None  # or: (sweep, impulse, tp, sl)

SWEEP_RANGE = [2, 3, 4, 5]
IMPULSE_RANGE = [4, 6, 8, 10]
# Wider targets, wider stop (less frantic than TP 8 / SL 3)
TP_RANGE = [10, 12, 15]
SL_RANGE = [4, 5, 6]

# --- Trend: EMA stack + slope must agree on ALL listed timeframes (MTF) ---
USE_TREND_FILTER = True
# Native 45m does not exist in MT5; M30 is the usual middle step between 15m and 1H.
TREND_MTF_FRAMES = [
    mt5.TIMEFRAME_M1,
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15,
    mt5.TIMEFRAME_M30,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_D1,
]
EMA_FAST_PERIOD = 21
EMA_SLOW_PERIOD = 50
EMA_SLOPE_LOOKBACK = 5
# Auto-extended when D1 (or other high-TF) is in TREND_MTF_FRAMES
TREND_WARMUP_EXTRA_DAYS_IF_D1 = 460

# --- Session filter (broker server clock hour, see offset below) ---
USE_SESSION_FILTER = True
# Add to UTC tick time so (hour) matches your MT5 server clock (e.g. EET often +2 or +3).
SERVER_HOUR_OFFSET_FROM_UTC = 0
# (start_hour, end_hour): end is exclusive. (7, 11) => 07:00–10:59. Overnight: (22, 6).
# Empty list = no session filter. Full day: [(0, 24)].
SESSION_WINDOWS = [
    (7, 12),
    (13, 18),
]

_TF_SECONDS = {
    mt5.TIMEFRAME_M1: 60,
    mt5.TIMEFRAME_M5: 300,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_M30: 1800,
    mt5.TIMEFRAME_H1: 3600,
    mt5.TIMEFRAME_H4: 14400,
    mt5.TIMEFRAME_D1: 86400,
}
_TF_NAMES = {
    mt5.TIMEFRAME_M1: "M1",
    mt5.TIMEFRAME_M5: "M5",
    mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M30: "M30",
    mt5.TIMEFRAME_H1: "H1",
    mt5.TIMEFRAME_H4: "H4",
    mt5.TIMEFRAME_D1: "D1",
}

MIN_TRADES_FOR_BEST = 50

RESET_RANGE_ON_TRADE_CLOSE = True
COOLDOWN_TICKS_AFTER_CLOSE = 50

SUMMARY_OUT = "best_strategy.csv"
TRADES_OUT = "trades.csv"

# ----------------------------
# SYMBOL & ACCOUNT (MT5)
# ----------------------------
_sym = mt5.symbol_info(symbol)
if _sym is None:
    print("symbol_info failed — is MT5 running and symbol valid?")
    raise SystemExit(1)

CONTRACT_SIZE = float(_sym.trade_contract_size)
POINT = float(_sym.point)
VOL_MIN = float(_sym.volume_min)
VOL_MAX = float(_sym.volume_max)
VOL_STEP = float(_sym.volume_step)
SYMBOL_DIGITS = int(_sym.digits)

_acc = mt5.account_info()
if _acc is not None:
    _live_balance = float(_acc.balance)
    ACCOUNT_CURRENCY = _acc.currency
else:
    _live_balance = ACCOUNT_BALANCE_FALLBACK
    ACCOUNT_CURRENCY = "USD"

if SIMULATE_ACCOUNT_BALANCE is not None:
    ACCOUNT_BALANCE = float(SIMULATE_ACCOUNT_BALANCE)
else:
    ACCOUNT_BALANCE = _live_balance


def estimated_margin_notional(price: float, lots: float) -> float:
    """Rough margin in account ccy (notional / leverage) if MT5 margin is 0."""
    ai = mt5.account_info()
    lev = float(ai.leverage) if ai and ai.leverage else 0.0
    if lev <= 0 or lev > 1_000_000:
        lev = float(ASSUMED_LEVERAGE)
    return (price * lots * CONTRACT_SIZE) / lev


def _spread_price_width():
    """Bid–ask width in price units (best effort from MT5)."""
    t = mt5.symbol_info_tick(symbol)
    if t is not None and t.bid and t.ask:
        return max(0.0, float(t.ask) - float(t.bid))
    sp = int(getattr(_sym, "spread", 0) or 0)
    return sp * POINT if sp else 0.0


def clamp_volume(lots: float) -> float:
    lots = max(VOL_MIN, min(VOL_MAX, lots))
    if VOL_STEP <= 0:
        return round(lots, 8)
    n = round((lots - VOL_MIN) / VOL_STEP)
    return round(VOL_MIN + n * VOL_STEP, 8)


def volume_for_trade(sl_price_distance: float) -> float:
    """Lot size for this parameter set and sizing mode."""
    if SIZING_MODE == "fixed":
        return clamp_volume(FIXED_LOT)
    # risk: size lots so MT5-reported loss at SL ≈ RISK_PERCENT of balance
    if sl_price_distance <= 0:
        return clamp_volume(VOL_MIN)
    risk_money = ACCOUNT_BALANCE * (RISK_PERCENT / 100.0)
    loss_1 = loss_per_lot_at_sl(sl_price_distance)
    if loss_1 <= 0:
        return clamp_volume(VOL_MIN)
    return clamp_volume(risk_money / loss_1)


def spread_cost_account_ccy(lots: float) -> float:
    """Rough round-trip spread cost using MT5 tick value (account currency)."""
    w = _spread_price_width()
    if w <= 0 or lots <= 0:
        return 0.0
    ts = float(_sym.trade_tick_size or 0)
    tv = float(_sym.trade_tick_value or 0)
    if ts > 0 and tv > 0:
        return (w / ts) * tv * lots
    return w * CONTRACT_SIZE * lots


def loss_per_lot_at_sl(sl_move: float) -> float:
    """|P&L| for 1.0 lot if price moves sl_move against a long (MT5-native)."""
    if sl_move <= 0:
        return 0.0
    ref = 2500.0
    r = mt5.order_calc_profit(
        mt5.ORDER_TYPE_BUY, symbol, 1.0, ref, ref - sl_move,
    )
    if r is None:
        return sl_move * CONTRACT_SIZE
    return abs(float(r))


def gross_profit_mt5(direction: str, entry: float, exit: float, lots: float) -> float:
    """Broker-accurate move P&L in account currency (no spread/commission)."""
    typ = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    r = mt5.order_calc_profit(typ, symbol, lots, entry, exit)
    if r is None:
        diff = (exit - entry) if direction == "BUY" else (entry - exit)
        return diff * CONTRACT_SIZE * lots
    return float(r)


def money_per_trade(
    direction: str, entry: float, exit: float, lots: float
) -> Tuple[float, float]:
    """
    (gross_move_account_ccy, spread_commission_deduction).
    """
    gross = gross_profit_mt5(direction, entry, exit, lots)
    deduction = 0.0
    if DEDUCT_SPREAD_EACH_TRADE:
        deduction += spread_cost_account_ccy(lots)
    deduction += COMMISSION_PER_LOT_ROUNDTRIP * lots
    return gross, deduction


# ----------------------------
# DATA
# ----------------------------
end_time = datetime.now()
start_time = end_time - timedelta(days=200)

ticks_data = mt5.copy_ticks_range(
    symbol,
    start_time,
    end_time,
    mt5.COPY_TICKS_ALL,
)

if ticks_data is None or len(ticks_data) == 0:
    print("No tick data — check MT5 connection, symbol, and history.")
    raise SystemExit(1)


def ema_series(closes: list, period: int) -> list:
    if not closes:
        return []
    k = 2.0 / (period + 1)
    out = [closes[0]]
    ema = closes[0]
    for i in range(1, len(closes)):
        ema = closes[i] * k + ema * (1.0 - k)
        out.append(ema)
    return out


def build_tick_trend_masks(
    tick_times: list,
    rates,
    ema_fast_n: int,
    ema_slow_n: int,
    slope_lookback: int,
    bar_sec: int,
):
    """
    Per tick: last fully closed bar at/before tick time must show
    long trend (fast>slow, fast EMA rising) or short (fast<slow, falling).
    No lookahead — only bars closed before the tick.
    """
    if rates is None or len(rates) < ema_slow_n + slope_lookback + 2:
        return None, None

    closes = [float(r["close"]) for r in rates]
    t_open = [int(r["time"]) for r in rates]
    t_end = [ot + bar_sec for ot in t_open]

    ema_f = ema_series(closes, ema_fast_n)
    ema_s = ema_series(closes, ema_slow_n)

    nbar = len(rates)
    long_ok_bar = [False] * nbar
    short_ok_bar = [False] * nbar
    min_i = max(ema_slow_n, slope_lookback, 1)
    for i in range(min_i, nbar):
        stack_long = ema_f[i] > ema_s[i]
        stack_short = ema_f[i] < ema_s[i]
        prev = i - slope_lookback
        slope = ema_f[i] - ema_f[prev]
        long_ok_bar[i] = stack_long and slope > 0
        short_ok_bar[i] = stack_short and slope < 0

    long_m = [False] * len(tick_times)
    short_m = [False] * len(tick_times)
    bar_i = -1
    for j, tick_t in enumerate(tick_times):
        while bar_i + 1 < nbar and t_end[bar_i + 1] <= tick_t:
            bar_i += 1
        if bar_i < min_i:
            continue
        long_m[j] = long_ok_bar[bar_i]
        short_m[j] = short_ok_bar[bar_i]

    return long_m, short_m


def combine_trend_masks(
    long_masks: list,
    short_masks: list,
) -> Optional[Tuple[list, list]]:
    if not long_masks or not short_masks:
        return None
    n = len(long_masks[0])
    long_out = [True] * n
    short_out = [True] * n
    for lm in long_masks:
        for j in range(n):
            long_out[j] = long_out[j] and lm[j]
    for sm in short_masks:
        for j in range(n):
            short_out[j] = short_out[j] and sm[j]
    return long_out, short_out


def trend_warmup_from(start: datetime, tfs: list) -> datetime:
    if any(tf == mt5.TIMEFRAME_D1 for tf in tfs):
        return start - timedelta(days=TREND_WARMUP_EXTRA_DAYS_IF_D1)
    return start - timedelta(days=300)


def build_mtf_trend_masks(
    tick_times: list,
    timeframes: list,
    warm_from: datetime,
    end_t: datetime,
):
    longs = []
    shorts = []
    for tf in timeframes:
        bar_sec = _TF_SECONDS.get(tf)
        if bar_sec is None:
            print(f"Unknown timeframe constant {tf}, skipping.")
            continue
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 2000)
        L, S = build_tick_trend_masks(
            tick_times,
            rates,
            EMA_FAST_PERIOD,
            EMA_SLOW_PERIOD,
            EMA_SLOPE_LOOKBACK,
            bar_sec,
        )
        if L is None:
            print(
    f"FAILED TF: {_TF_NAMES.get(tf)} | bars={len(rates) if rates is not None else 0}"
)
            return None, None
        longs.append(L)
        shorts.append(S)
    return combine_trend_masks(longs, shorts)


def hour_on_server(ts: int) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(
        hours=SERVER_HOUR_OFFSET_FROM_UTC
    )
    return dt.hour


def in_session(hour: int, windows: list) -> bool:
    if not windows:
        return True
    for a, b in windows:
        if a <= b:
            if a <= hour < b:
                return True
        else:
            if hour >= a or hour < b:
                return True
    return False


def build_session_mask(tick_times: list) -> Optional[list]:
    if not USE_SESSION_FILTER or not SESSION_WINDOWS:
        return None
    return [in_session(hour_on_server(ts), SESSION_WINDOWS) for ts in tick_times]


def tick_mid(t):
    b = float(t["bid"])
    try:
        a = float(t["ask"])
        if a > 0:
            return (a + b) / 2
    except (KeyError, TypeError, ValueError):
        pass
    return b


price_series = [tick_mid(t) for t in ticks_data]
tick_times = [int(t["time"]) for t in ticks_data]

TREND_LONG_MASK = None
TREND_SHORT_MASK = None
SESSION_MASK = None
MTF_LABEL = ""

if USE_TREND_FILTER and TREND_MTF_FRAMES:
    warm_from = trend_warmup_from(start_time, TREND_MTF_FRAMES)
    TREND_LONG_MASK, TREND_SHORT_MASK = build_mtf_trend_masks(
        tick_times,
        list(TREND_MTF_FRAMES),
        warm_from,
        end_time,
    )
    if TREND_LONG_MASK is None:
        print("MTF trend filter: failed — running without trend filter.")
        TREND_LONG_MASK, TREND_SHORT_MASK = None, None
    else:
        MTF_LABEL = ",".join(_TF_NAMES.get(tf, str(tf)) for tf in TREND_MTF_FRAMES)
        print(
            f"MTF trend ON: EMA {EMA_FAST_PERIOD}/{EMA_SLOW_PERIOD}, "
            f"slope={EMA_SLOPE_LOOKBACK} bars; TFs=[{MTF_LABEL}]"
        )
elif USE_TREND_FILTER and not TREND_MTF_FRAMES:
    print("USE_TREND_FILTER is True but TREND_MTF_FRAMES is empty — no trend filter.")

SESSION_MASK = build_session_mask(tick_times)
if USE_SESSION_FILTER and SESSION_WINDOWS:
    print(
        f"Session filter ON: windows={SESSION_WINDOWS} "
        f"server_hour_utc_offset={SERVER_HOUR_OFFSET_FROM_UTC}"
    )
elif USE_SESSION_FILTER and not SESSION_WINDOWS:
    print("Session filter requested but SESSION_WINDOWS empty — all hours allowed.")
print(f"Loaded ticks: {len(price_series)}")
print(
    f"MT5: contract_size={CONTRACT_SIZE}  point={POINT}  "
    f"volume {VOL_MIN}–{VOL_MAX} step {VOL_STEP}  "
    f"balance≈{ACCOUNT_BALANCE:.2f} {ACCOUNT_CURRENCY}"
)
if SIMULATE_ACCOUNT_BALANCE is not None:
    print(
        f"  (SIMULATE_ACCOUNT_BALANCE override; MT5 wallet was {_live_balance:.2f})"
    )
print(f"Sizing: {SIZING_MODE}  (risk {RISK_PERCENT}% per trade)")

spread_w = _spread_price_width()
print(f"Spread width (now): ~{spread_w:.{SYMBOL_DIGITS}f} price units")


def run_backtest(sweep, impulse, tp, sl, trade_rows_out=None):
    lots = volume_for_trade(float(sl))

    position = None
    state = "IDLE"
    cooldown = 0

    trades = wins = losses = 0
    net_points = 0.0
    net_money = 0.0
    win_sum_pt = loss_sum_pt = 0.0
    skipped_margin = 0
    initial_balance = float(ACCOUNT_BALANCE)
    equity = initial_balance
    peak = initial_balance
    max_dd = 0.0

    high = low = price_series[0]

    for j, price in enumerate(price_series):
        ok_long = (
            (TREND_LONG_MASK[j] if TREND_LONG_MASK is not None else True)
            and (SESSION_MASK[j] if SESSION_MASK is not None else True)
        )
        ok_short = (
            (TREND_SHORT_MASK[j] if TREND_SHORT_MASK is not None else True)
            and (SESSION_MASK[j] if SESSION_MASK is not None else True)
        )

        if price > high:
            high = price
        if price < low:
            low = price

        if position is not None:
            direction, entry = position
            pnl = price - entry if direction == "BUY" else entry - price

            if pnl >= tp:
                wins += 1
                trades += 1
                net_points += pnl
                win_sum_pt += pnl
                g, ded = money_per_trade(direction, entry, price, lots)
                cash = g - ded
                net_money += cash
                equity += cash
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
                if trade_rows_out is not None:
                    trade_rows_out.append(
                        [
                            direction,
                            entry,
                            price,
                            pnl,
                            lots,
                            g,
                            ded,
                            cash,
                            "TP",
                        ]
                    )
                position = None
                if RESET_RANGE_ON_TRADE_CLOSE:
                    high = low = price
                cooldown = COOLDOWN_TICKS_AFTER_CLOSE

            elif pnl <= -sl:
                losses += 1
                trades += 1
                net_points += pnl
                loss_sum_pt += pnl
                g, ded = money_per_trade(direction, entry, price, lots)
                cash = g - ded
                net_money += cash
                equity += cash
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
                if trade_rows_out is not None:
                    trade_rows_out.append(
                        [
                            direction,
                            entry,
                            price,
                            pnl,
                            lots,
                            g,
                            ded,
                            cash,
                            "SL",
                        ]
                    )
                position = None
                if RESET_RANGE_ON_TRADE_CLOSE:
                    high = low = price
                cooldown = COOLDOWN_TICKS_AFTER_CLOSE

            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        if state == "IDLE":
            if price <= low + sweep:
                state = "SWEEP_DOWN"
            elif price >= high - sweep:
                state = "SWEEP_UP"

        if state == "SWEEP_DOWN":
            if price - low >= impulse and ok_long:
                if equity <= 0:
                    continue
                margin_need = estimated_margin_notional(price, lots) * MARGIN_BUFFER
                if margin_need > 0 and equity < margin_need:
                    skipped_margin += 1
                    continue
                position = ("BUY", price)
                state = "IDLE"

        elif state == "SWEEP_UP":
            if high - price >= impulse and ok_short:
                if equity <= 0:
                    continue
                margin_need = estimated_margin_notional(price, lots) * MARGIN_BUFFER
                if margin_need > 0 and equity < margin_need:
                    skipped_margin += 1
                    continue
                position = ("SELL", price)
                state = "IDLE"

    winrate = (wins / trades * 100) if trades > 0 else 0
    avg_win = (win_sum_pt / wins) if wins > 0 else 0
    avg_loss = (loss_sum_pt / losses) if losses > 0 else 0
    exp_pt = (net_points / trades) if trades > 0 else 0
    exp_m = (net_money / trades) if trades > 0 else 0

    loss_1lot_full_sl = loss_per_lot_at_sl(float(sl))
    risk_if_full_sl = loss_1lot_full_sl * lots
    ending_balance = equity
    target_r = (
        ACCOUNT_BALANCE * (RISK_PERCENT / 100.0) if SIZING_MODE == "risk" else None
    )
    risk_bumped_by_min_lot = (
        target_r is not None and risk_if_full_sl > target_r * 1.05 and lots <= VOL_MIN + 1e-12
    )

    return {
        "sweep": sweep,
        "impulse": impulse,
        "tp": tp,
        "sl": sl,
        "lots": lots,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "net_points": net_points,
        "net_money": net_money,
        "starting_balance": initial_balance,
        "ending_balance": ending_balance,
        "skipped_margin_entries": skipped_margin,
        "risk_bumped_by_min_lot": risk_bumped_by_min_lot,
        "max_drawdown_money": max_dd,
        "avg_win_pts": avg_win,
        "avg_loss_pts": avg_loss,
        "expectancy_points": exp_pt,
        "expectancy_money": exp_m,
        "risk_if_full_sl_hit": risk_if_full_sl,
        "target_risk_dollars": target_r,
        "account_balance": ACCOUNT_BALANCE,
        "risk_percent": RISK_PERCENT if SIZING_MODE == "risk" else None,
        "sizing_mode": SIZING_MODE,
        "use_trend_filter": USE_TREND_FILTER and TREND_LONG_MASK is not None,
        "mtf_timeframes": MTF_LABEL,
        "use_session_filter": SESSION_MASK is not None,
        "session_windows": str(SESSION_WINDOWS) if SESSION_WINDOWS else "",
        "server_hour_offset_utc": SERVER_HOUR_OFFSET_FROM_UTC,
        "ema_fast": EMA_FAST_PERIOD,
        "ema_slow": EMA_SLOW_PERIOD,
        "ema_slope_bars": EMA_SLOPE_LOOKBACK,
    }


def pick_best_params():
    all_metrics = []
    total = len(SWEEP_RANGE) * len(IMPULSE_RANGE) * len(TP_RANGE) * len(SL_RANGE)
    n = 0
    for s, i, tpv, slv in itertools.product(
        SWEEP_RANGE, IMPULSE_RANGE, TP_RANGE, SL_RANGE,
    ):
        n += 1
        if n % 64 == 0 or n == 1:
            print(f"  search [{n}/{total}] …")
        all_metrics.append(run_backtest(s, i, tpv, slv, trade_rows_out=None))

    candidates = [m for m in all_metrics if m["trades"] >= MIN_TRADES_FOR_BEST]
    if not candidates:
        print(
            f"No combo had >= {MIN_TRADES_FOR_BEST} trades; using full grid."
        )
        candidates = all_metrics

    viable = [m for m in candidates if m["net_money"] > 0]
    pool = viable if viable else candidates
    best = max(pool, key=lambda m: (m["net_money"], m["trades"]))
    if not viable:
        print(
            "Warning: no parameter set had positive net "
            f"({ACCOUNT_CURRENCY}) in this window; reporting best available."
        )
    return best


def write_outputs(metrics, trade_rows):
    with open(SUMMARY_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sweep", "impulse", "tp", "sl",
            "lots", "sizing_mode", "risk_pct", "balance_used",
            "use_trend_filter", "mtf_timeframes",
            "use_session_filter", "session_windows", "server_hour_offset_utc",
            "ema_fast", "ema_slow", "ema_slope_bars",
            "trades", "wins", "losses", "winrate",
            "net_points", f"net_{ACCOUNT_CURRENCY.lower()}",
            "expectancy_points", f"expectancy_{ACCOUNT_CURRENCY.lower()}",
            "max_drawdown_money", "risk_if_full_sl_hit",
            "target_risk_dollars",
            "starting_balance", "ending_balance", "skipped_margin_entries",
            "risk_bumped_by_min_lot",
            "contract_size", "commission_per_lot_rt",
        ])
        w.writerow([
            metrics["sweep"], metrics["impulse"], metrics["tp"], metrics["sl"],
            metrics["lots"], metrics["sizing_mode"], metrics["risk_percent"],
            metrics["account_balance"],
            metrics["use_trend_filter"], metrics["mtf_timeframes"],
            metrics["use_session_filter"], metrics["session_windows"],
            metrics["server_hour_offset_utc"],
            metrics["ema_fast"], metrics["ema_slow"],
            metrics["ema_slope_bars"],
            metrics["trades"], metrics["wins"], metrics["losses"],
            metrics["winrate"],
            metrics["net_points"], metrics["net_money"],
            metrics["expectancy_points"], metrics["expectancy_money"],
            metrics["max_drawdown_money"], metrics["risk_if_full_sl_hit"],
            metrics["target_risk_dollars"],
            metrics["starting_balance"], metrics["ending_balance"],
            metrics["skipped_margin_entries"],
            metrics["risk_bumped_by_min_lot"],
            CONTRACT_SIZE, COMMISSION_PER_LOT_ROUNDTRIP,
        ])

    with open(TRADES_OUT, "w", newline="") as f:
        tw = csv.writer(f)
        tw.writerow([
            "direction", "entry", "exit", "pnl_points", "lots",
            "gross_symbol_ccy", "deductions", f"net_{ACCOUNT_CURRENCY.lower()}",
            "reason",
        ])
        for row in trade_rows:
            tw.writerow(row)


# ----------------------------
# RUN
# ----------------------------
if FIXED_PARAMS is not None:
    s, i, tpv, slv = FIXED_PARAMS
    print(f"Using fixed params: sweep={s} impulse={i} tp={tpv} sl={slv}")
    rows = []
    metrics = run_backtest(s, i, tpv, slv, trade_rows_out=rows)
else:
    print(
        f"Searching grid (min {MIN_TRADES_FOR_BEST} trades) "
        f"for best net ({ACCOUNT_CURRENCY})…"
    )
    best = pick_best_params()
    print(
        f"Best: sweep={best['sweep']} impulse={best['impulse']} "
        f"tp={best['tp']} sl={best['sl']}  lots={best['lots']}"
    )
    rows = []
    metrics = run_backtest(
        best["sweep"],
        best["impulse"],
        best["tp"],
        best["sl"],
        trade_rows_out=rows,
    )

write_outputs(metrics, rows)

print("")
print("--- Single strategy (account-sized) ---")
print(
    f"Params:  sweep={metrics['sweep']}  impulse={metrics['impulse']}  "
    f"TP={metrics['tp']}  SL={metrics['sl']}  |  "
    f"trend={metrics['use_trend_filter']}  session={metrics['use_session_filter']}"
)
print(
    f"Balance model: start={metrics['starting_balance']:.2f}  "
    f"end={metrics['ending_balance']:.2f}  "
    f"(sum of closed trades = {metrics['net_money']:.2f})"
)
if metrics.get("skipped_margin_entries", 0) > 0:
    print(
        f"Skipped {metrics['skipped_margin_entries']} entries: equity < margin×{MARGIN_BUFFER} "
        f"(~notional/{ASSUMED_LEVERAGE} if broker leverage missing)."
    )
if metrics.get("risk_bumped_by_min_lot"):
    print(
        "Warning: min lot forces risk ABOVE your RISK_PERCENT target "
        "(raise balance or use a broker with smaller min volume)."
    )
print(f"Volume:  {metrics['lots']} lot  |  contract {CONTRACT_SIZE} oz/lot")
if metrics.get("mtf_timeframes"):
    print(f"MTF:     {metrics['mtf_timeframes']}")
if metrics.get("session_windows"):
    print(
        f"Sessions: {metrics['session_windows']}  "
        f"(UTC offset {metrics['server_hour_offset_utc']}h → match MT5 server clock)"
    )
print(
    f"If full SL hits: ~{metrics['risk_if_full_sl_hit']:.2f} {ACCOUNT_CURRENCY} "
    f"({metrics['lots']} lot × loss/lot @ SL)"
)
if metrics["target_risk_dollars"] is not None:
    print(
        f"Target risk (sizing): {metrics['target_risk_dollars']:.2f} "
        f"({RISK_PERCENT}% of balance; lot rounded to step)"
    )
print(
    f"Trades:  {metrics['trades']}  "
    f"(wins {metrics['wins']} / losses {metrics['losses']})"
)
print(f"Win %:   {metrics['winrate']:.2f}")
print(
    "(Low win % with TP > SL is common; focus on net & expectancy for demo validation.)"
)
print(f"Net (price): {metrics['net_points']:.4f} pts  (~$/oz before costs × size)")
print(
    f"Net ({ACCOUNT_CURRENCY}): {metrics['net_money']:.2f}  "
    f"(after spread{' + comm' if COMMISSION_PER_LOT_ROUNDTRIP else ''})"
)
print(f"Max DD ({ACCOUNT_CURRENCY}): {metrics['max_drawdown_money']:.2f}")
print(f"Avg per trade: {metrics['expectancy_money']:.2f} {ACCOUNT_CURRENCY}")
print("")
print(f"Wrote {SUMMARY_OUT} and {TRADES_OUT}")
