"""
Shared MetaTrader5 helpers for server.py and trader.py.
One MT5 terminal ↔ one Python process at a time.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import MetaTrader5 as mt5

SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2


def ensure_mt5() -> Tuple[bool, bool]:
    """
    Connect to the terminal if needed.
    Returns (ok, we_called_initialize).
    """
    if mt5.terminal_info() is not None:
        return True, False
    if not mt5.initialize():
        return False, False
    return True, True


def shutdown_mt5_if_owned(we_initialized: bool) -> None:
    if we_initialized:
        mt5.shutdown()


def symbols_catalog() -> list:
    syms = mt5.symbols_get()
    if syms is None:
        return []
    return list(syms)


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
    for sym in symbols_catalog():
        if sym.name.upper() == needle:
            return sym.name

    # Prefix match: XAUUSD / XAUUSDz / GOLD → broker-specific name
    alnum = "".join(c for c in s if c.isalnum()).upper()
    if len(alnum) < 4:
        return None
    for length in range(len(alnum), 3, -1):
        prefix = alnum[:length]
        matches = [sym.name for sym in symbols_catalog() if sym.name.upper().startswith(prefix)]
        if not matches:
            continue
        if len(matches) == 1:
            return matches[0]
        picked = _pick_best_symbol(matches, preferred=needle)
        if picked:
            return picked
    return None


def _pick_best_symbol(names: List[str], preferred: str) -> Optional[str]:
    for name in names:
        if name.upper() == preferred:
            return name
    visible = [n for n in names if (mt5.symbol_info(n) or None) and mt5.symbol_info(n).visible]
    if visible:
        return sorted(visible, key=len)[0]
    for name in sorted(names, key=len):
        info = mt5.symbol_info(name)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(name, True)
            return info.name
    return names[0] if names else None


def default_trader_symbol() -> str:
    """Default instrument (override with TRADER_SYMBOL env)."""
    return os.environ.get("TRADER_SYMBOL", "XAUUSD").strip()


def is_xau_symbol(symbol: str) -> bool:
    u = (symbol or "").upper()
    return "XAU" in u or "GOLD" in u


def apply_symbol_tuning(cfg: dict, symbol: str, aggression_drift_pct: float) -> None:
    """
    Adjust spread/drift/cluster for gold vs crypto after symbol is resolved.
    Called from trader.py once the broker's exact symbol name is known.
    """
    if is_xau_symbol(symbol):
        cfg["cluster_pct"] = 0.0012
        cfg["max_spread_pct"] = 0.0005
        cfg["max_entry_drift_pct"] = aggression_drift_pct * 2.0
        cfg["sl_buffer_ticks"] = 5
        cfg["trend_pullback_pct"] = 0.0015
    else:
        cfg["cluster_pct"] = 0.0015
        cfg["max_spread_pct"] = 0.003
        cfg["max_entry_drift_pct"] = aggression_drift_pct
        cfg["sl_buffer_ticks"] = 3
        cfg["trend_pullback_pct"] = 0.0012


def mt5_busy_hint() -> str:
    return (
        "MT5 may already be connected to another Python process (e.g. uvicorn server.py). "
        "Stop that process, or run the trader inside the server: set ENABLE_TRADER=1 "
        "and restart uvicorn."
    )
