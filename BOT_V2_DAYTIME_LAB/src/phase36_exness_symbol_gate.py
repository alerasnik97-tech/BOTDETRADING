from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase36r_37a_micro_real_gate" / "exness_symbol_gate"


def _load_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except Exception:
        return None


def get_exness_symbol_gate_status(symbol_candidates: list[str] | None = None, max_spread_pips: float = 1.2) -> dict[str, Any]:
    symbol_candidates = symbol_candidates or ["EURUSD", "EURUSDm", "EURUSD.raw", "EURUSDc"]
    status: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": "ERROR_FAIL_CLOSED",
        "symbol_detected": None,
        "spread_pips": None,
        "min_lot": None,
        "max_lot": None,
        "lot_step": None,
        "digits": None,
        "point": None,
        "tick_size": None,
        "contract_size": None,
        "stops_level": None,
        "freeze_level": None,
        "trade_mode": None,
        "filling_mode": None,
        "order_mode": None,
        "margin_initial": None,
        "reason": "",
    }
    mt5 = _load_mt5()
    if mt5 is None:
        status["state"] = "NO_TRADE_SYMBOL_NOT_FOUND"
        status["reason"] = "MetaTrader5 Python module unavailable"
        return status
    try:
        if not mt5.initialize():
            status["state"] = "NO_TRADE_SYMBOL_NOT_FOUND"
            status["reason"] = "MT5 terminal not initialized/connected"
            return status
        info = None
        for candidate in symbol_candidates:
            info = mt5.symbol_info(candidate)
            if info is not None:
                status["symbol_detected"] = candidate
                break
        if info is None:
            status["state"] = "NO_TRADE_SYMBOL_NOT_FOUND"
            status["reason"] = "No EURUSD/Exness suffix symbol found"
            return status
        tick = mt5.symbol_info_tick(status["symbol_detected"])
        bid = float(getattr(tick, "bid", 0.0) or 0.0) if tick is not None else 0.0
        ask = float(getattr(tick, "ask", 0.0) or 0.0) if tick is not None else 0.0
        point = float(getattr(info, "point", 0.00001) or 0.00001)
        pip = point * 10 if point < 0.001 else point
        spread = (ask - bid) / pip if ask > bid > 0 else None
        status.update({
            "spread_pips": round(spread, 3) if spread is not None else None,
            "min_lot": float(getattr(info, "volume_min", 0.0) or 0.0),
            "max_lot": float(getattr(info, "volume_max", 0.0) or 0.0),
            "lot_step": float(getattr(info, "volume_step", 0.0) or 0.0),
            "digits": int(getattr(info, "digits", 0) or 0),
            "point": point,
            "tick_size": float(getattr(info, "trade_tick_size", 0.0) or 0.0),
            "contract_size": float(getattr(info, "trade_contract_size", 0.0) or 0.0),
            "stops_level": int(getattr(info, "trade_stops_level", 0) or 0),
            "freeze_level": int(getattr(info, "trade_freeze_level", 0) or 0),
            "trade_mode": int(getattr(info, "trade_mode", 0) or 0),
            "filling_mode": int(getattr(info, "filling_mode", 0) or 0),
            "order_mode": int(getattr(info, "order_mode", 0) or 0),
            "margin_initial": float(getattr(info, "margin_initial", 0.0) or 0.0),
        })
        if spread is None:
            status["state"] = "NO_TRADE_SPREAD_TOO_HIGH"
            status["reason"] = "Spread missing"
            return status
        if spread > max_spread_pips:
            status["state"] = "NO_TRADE_SPREAD_TOO_HIGH"
            status["reason"] = f"Spread {spread:.3f} pips > {max_spread_pips}"
            return status
        if status["trade_mode"] <= 0:
            status["state"] = "NO_TRADE_SYMBOL_NOT_FOUND"
            status["reason"] = "Trade mode unavailable"
            return status
        status["state"] = "ALLOW"
        status["reason"] = "Symbol/spread/stoplevel gate passed"
        return status
    except Exception as exc:
        status["state"] = "ERROR_FAIL_CLOSED"
        status["reason"] = str(exc)
        return status


def write_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    status = get_exness_symbol_gate_status()
    (OUT / "phase36r_exness_symbol_gate.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    md = [
        "# Phase36R Exness Symbol Gate",
        "",
        f"- state: {status['state']}",
        f"- symbol_detected: {status['symbol_detected']}",
        f"- spread_pips: {status['spread_pips']}",
        f"- min_lot: {status['min_lot']}",
        f"- lot_step: {status['lot_step']}",
        f"- stops_level: {status['stops_level']}",
        f"- reason: {status['reason']}",
    ]
    (OUT / "phase36r_exness_symbol_gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2))
