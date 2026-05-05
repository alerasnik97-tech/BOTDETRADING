from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase36s_live_news_lot_feasibility" / "server_time_validation"
NY = ZoneInfo("America/New_York")
AR = ZoneInfo("America/Argentina/Buenos_Aires")


def _load_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except Exception:
        return None


def validate_server_time(symbol_candidates: list[str] | None = None, max_tick_age_seconds: int = 120) -> dict[str, Any]:
    symbol_candidates = symbol_candidates or ["EURUSD", "EURUSDm", "EURUSD.raw", "EURUSDc"]
    now_utc = datetime.now(timezone.utc)
    status: dict[str, Any] = {
        "timestamp_utc": now_utc.isoformat(),
        "local_time": datetime.now().astimezone().isoformat(),
        "utc_time": now_utc.isoformat(),
        "ny_time": now_utc.astimezone(NY).isoformat(),
        "argentina_time": now_utc.astimezone(AR).isoformat(),
        "weekday_ny": now_utc.astimezone(NY).strftime("%A"),
        "ny_dst_active": bool(now_utc.astimezone(NY).dst()),
        "mt5_available": False,
        "terminal_initialized": False,
        "symbol": None,
        "last_tick_time_utc": None,
        "tick_age_seconds": None,
        "server_vs_utc_seconds": None,
        "server_vs_ny_seconds": None,
        "operating_window_ny": "07:00-16:30",
        "state": "ERROR_FAIL_CLOSED",
        "reason": "",
    }
    mt5 = _load_mt5()
    if mt5 is None:
        status["state"] = "MICRO_REAL_BLOCKED_SERVER_TIME"
        status["reason"] = "MetaTrader5 Python module unavailable"
        return status
    status["mt5_available"] = True
    try:
        if not mt5.initialize():
            status["state"] = "MICRO_REAL_BLOCKED_SERVER_TIME"
            status["reason"] = "MT5 terminal not initialized"
            return status
        status["terminal_initialized"] = True
        selected = None
        for candidate in symbol_candidates:
            if mt5.symbol_info(candidate) is not None:
                selected = candidate
                break
        if selected is None:
            status["state"] = "MICRO_REAL_BLOCKED_SERVER_TIME"
            status["reason"] = "No EURUSD symbol available for tick-time validation"
            return status
        status["symbol"] = selected
        tick = mt5.symbol_info_tick(selected)
        if tick is None or not getattr(tick, "time", 0):
            status["state"] = "MICRO_REAL_BLOCKED_SERVER_TIME"
            status["reason"] = "No tick timestamp available"
            return status
        tick_utc = datetime.fromtimestamp(int(tick.time), tz=timezone.utc)
        age = (now_utc - tick_utc).total_seconds()
        status["last_tick_time_utc"] = tick_utc.isoformat()
        status["tick_age_seconds"] = round(age, 3)
        status["server_vs_utc_seconds"] = round((tick_utc - now_utc).total_seconds(), 3)
        status["server_vs_ny_seconds"] = round((tick_utc - now_utc.astimezone(NY).astimezone(timezone.utc)).total_seconds(), 3)
        if age < 0:
            status["state"] = "MICRO_REAL_BLOCKED_SERVER_TIME"
            status["reason"] = "Tick timestamp is in the future"
            return status
        if age > max_tick_age_seconds:
            status["state"] = "MICRO_REAL_BLOCKED_SERVER_TIME"
            status["reason"] = f"Tick timestamp stale: {age:.1f}s"
            return status
        status["state"] = "ALLOW"
        status["reason"] = "MT5 tick timestamp is current enough for UTC/NY gate alignment"
        return status
    except Exception as exc:
        status["state"] = "ERROR_FAIL_CLOSED"
        status["reason"] = str(exc)
        return status


def write_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    status = validate_server_time()
    (OUT / "phase36s_server_time_validation.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    md = [
        "# Phase36S Server Time Validation",
        "",
        f"- state: {status['state']}",
        f"- symbol: {status['symbol']}",
        f"- UTC: {status['utc_time']}",
        f"- NY: {status['ny_time']}",
        f"- last_tick_time_utc: {status['last_tick_time_utc']}",
        f"- tick_age_seconds: {status['tick_age_seconds']}",
        f"- server_vs_utc_seconds: {status['server_vs_utc_seconds']}",
        f"- reason: {status['reason']}",
    ]
    (OUT / "phase36s_server_time_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2))
