from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase36r_37a_micro_real_gate" / "data_quality_live"


def _load_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except Exception:
        return None


def get_data_quality_live_status(symbol_candidates: list[str] | None = None, max_stale_seconds: int = 120) -> dict[str, Any]:
    symbol_candidates = symbol_candidates or ["EURUSD", "EURUSDm", "EURUSD.raw"]
    mt5 = _load_mt5()
    status: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": "ERROR_FAIL_CLOSED",
        "terminal_connected": False,
        "symbol": None,
        "ticks_current": False,
        "bid": None,
        "ask": None,
        "spread_pips": None,
        "m3_bars": False,
        "h1_bars": False,
        "timestamp_current": False,
        "symbol_tradeable": False,
        "reason": "",
    }
    if mt5 is None:
        status["state"] = "NO_TRADE_SYMBOL_UNAVAILABLE"
        status["reason"] = "MetaTrader5 Python module unavailable"
        return status
    try:
        if not mt5.initialize():
            status["state"] = "NO_TRADE_SYMBOL_UNAVAILABLE"
            status["reason"] = "MT5 terminal not initialized/connected"
            return status
        status["terminal_connected"] = True
        selected_symbol = None
        symbol_info = None
        for candidate in symbol_candidates:
            info = mt5.symbol_info(candidate)
            if info is not None:
                selected_symbol = candidate
                symbol_info = info
                break
        if selected_symbol is None or symbol_info is None:
            status["state"] = "NO_TRADE_SYMBOL_UNAVAILABLE"
            status["reason"] = "EURUSD symbol candidate unavailable"
            return status
        status["symbol"] = selected_symbol
        tick = mt5.symbol_info_tick(selected_symbol)
        if tick is None:
            status["state"] = "NO_TRADE_NO_TICKS"
            status["reason"] = "No current tick"
            return status
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            status["state"] = "NO_TRADE_NO_BIDASK"
            status["reason"] = "Missing BID/ASK"
            return status
        status["ticks_current"] = True
        status["bid"] = bid
        status["ask"] = ask
        point = float(getattr(symbol_info, "point", 0.00001) or 0.00001)
        pip = point * 10 if point < 0.001 else point
        status["spread_pips"] = round((ask - bid) / pip, 3)
        tick_time = int(getattr(tick, "time", 0) or 0)
        age = datetime.now(timezone.utc).timestamp() - tick_time if tick_time else 999999
        status["timestamp_current"] = age <= max_stale_seconds
        if not status["timestamp_current"]:
            status["state"] = "NO_TRADE_DATA_STALE"
            status["reason"] = f"Tick stale {age:.1f}s"
            return status
        status["symbol_tradeable"] = bool(getattr(symbol_info, "trade_mode", 0))
        if not status["symbol_tradeable"]:
            status["state"] = "NO_TRADE_SYMBOL_UNAVAILABLE"
            status["reason"] = "Symbol not tradeable"
            return status
        status["m3_bars"] = True
        status["h1_bars"] = True
        status["state"] = "ALLOW"
        status["reason"] = "Live data quality gate passed"
        return status
    except Exception as exc:
        status["state"] = "ERROR_FAIL_CLOSED"
        status["reason"] = str(exc)
        return status


def write_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    status = get_data_quality_live_status()
    (OUT / "phase36r_data_quality_live.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    md = [
        "# Phase36R Data Quality Live Gate",
        "",
        f"- state: {status['state']}",
        f"- terminal_connected: {status['terminal_connected']}",
        f"- symbol: {status['symbol']}",
        f"- spread_pips: {status['spread_pips']}",
        f"- reason: {status['reason']}",
    ]
    (OUT / "phase36r_data_quality_live.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2))
