# Start the live trader (MT5 must be open and logged in).
# Defaults: XAUUSD, aggression 8. Override with env vars below.

$env:TRADER_AGGRESSION = "8"
# $env:TRADER_SYMBOL = "XAUUSDz"   # uncomment if broker uses a suffix name

Set-Location $PSScriptRoot
python trader.py
