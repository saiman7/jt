# JournalIT MT5 API (`jt`)

FastAPI gateway to MetaTrader 5 for backtesting, live charts, liquidity levels, and M5 reversal prediction.

## Requirements

- MetaTrader 5 terminal installed and logged in (same machine as the API)
- Python 3.13+

## Install & run

```bash
cd jt
uv sync
uv run python main.py
```

Or:

```bash
cd jt
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Point the Next.js app at this server:

```env
NEXT_PUBLIC_LIVE_MT5_URL=http://127.0.0.1:8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | MT5 connection status |
| GET | `/api/historical` | Backtest candles + liquidity (`symbol`, `timeframe`, `start_time`, `end_time`) |
| GET | `/api/recent-candles` | Last N bars (`symbol`, `timeframe`, `count`) |
| GET | `/api/forming-candle` | Forming M5 bar from M1 stream (`symbol`, `chart_timeframe`) |
| GET | `/api/reversal-live` | Confirmed + predicted reversals from live MT5 |
| POST | `/api/reversal-scan` | Scan JSON candles + liquidity (no MT5 fetch) |
| GET | `/api/positions` | Open positions (`symbol`, `magic`) |
| POST | `/api/trade` | Market order with SL/TP |
| POST | `/api/lot-for-risk` | Lot size for target USD risk |
| POST | `/api/calc-profit` | P&L estimate |
| WS | `/ws/rates` | Streaming candles + liquidity; M1 also sends `reversalPredictions` |

## Reversal prediction (M1 chart)

When the chart timeframe is **M1**, confirmation uses **M5**:

- **Confirmed** — closed M5 reversal candle (`signals` in `/api/reversal-live` or WebSocket `reversalSignals`)
- **Predicted** — current forming M5 built from M1 ticks (`predictions` / `reversalPredictions`, `formingConfirmCandle`)

Patterns:

- **BUY** after SSL sweep: red → green (`close > open` and `low < open`)
- **SELL** after BSL sweep: green → red (`close < open` and `high > open`)

## Example

```bash
curl "http://127.0.0.1:8000/api/health"
curl "http://127.0.0.1:8000/api/reversal-live?symbol=BTCUSDr&chart_timeframe=M1"
curl "http://127.0.0.1:8000/api/forming-candle?symbol=BTCUSDr&chart_timeframe=M1"
```
