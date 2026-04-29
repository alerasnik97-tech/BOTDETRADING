from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_support import OUT, account_gate, ensure_mt5, write_json, write_text


PHASE_OUT = OUT.parent / "phase37c_full_auto_ftmo_trial_bootstrap"


def autodetect() -> dict[str, Any]:
    status = account_gate()
    result: dict[str, Any] = {
        "timestamp_utc": status["timestamp_utc"],
        "state": "BLOCKED_NO_MT5_CONNECTION",
        "terminal_connected": status.get("terminal_connected"),
        "terminal_path": None,
        "data_path": None,
        "commondata_path": None,
        "company": status.get("company"),
        "server": status.get("server"),
        "trade_mode": status.get("trade_mode"),
        "trade_mode_label": status.get("trade_mode_label"),
        "balance": status.get("balance"),
        "currency": status.get("currency"),
        "ftmo": False,
        "demo_trial": False,
        "eurusd_available": False,
        "reason": status.get("reason"),
    }
    if status.get("state") not in {"FTMO_DEMO_TRIAL_CONFIRMED", "WARNING_BALANCE_NOT_10K"}:
        result["state"] = status.get("state", "BLOCKED_NO_MT5_CONNECTION")
        return result
    mt5, error = ensure_mt5()
    if mt5 is None:
        result["state"] = "BLOCKED_NO_MT5_CONNECTION"
        result["reason"] = error
        return result
    term = mt5.terminal_info()
    if term is None:
        result["state"] = "BLOCKED_NO_DATA_PATH"
        result["reason"] = "terminal_info unavailable"
        return result
    term_dict = term._asdict() if hasattr(term, "_asdict") else {}
    result["terminal_path"] = term_dict.get("path")
    result["data_path"] = term_dict.get("data_path")
    result["commondata_path"] = term_dict.get("commondata_path")
    result["ftmo"] = "ftmo" in f"{result['company']} {result['server']}".lower()
    result["demo_trial"] = result["trade_mode_label"] in {"DEMO", "CONTEST"}
    result["eurusd_available"] = mt5.symbol_info("EURUSD") is not None
    if not result["ftmo"]:
        result["state"] = "BLOCKED_NOT_FTMO"
    elif not result["demo_trial"]:
        result["state"] = "BLOCKED_REAL_ACCOUNT_DETECTED"
    elif not result["data_path"]:
        result["state"] = "BLOCKED_NO_DATA_PATH"
    else:
        result["state"] = "FTMO_MT5_DEMO_AUTODETECTED"
        result["reason"] = "FTMO demo/trial terminal autodetected with data_path"
    return result


def write_outputs() -> dict[str, Any]:
    result = autodetect()
    write_json(PHASE_OUT / "mt5_autodetect" / "phase37c_mt5_autodetect.json", result)
    write_text(
        PHASE_OUT / "mt5_autodetect" / "phase37c_mt5_autodetect.md",
        f"""
# Phase37C MT5 Autodetect

- state: {result['state']}
- FTMO: {result['ftmo']}
- demo/trial: {result['demo_trial']}
- data_path: {result['data_path']}
- commondata_path: {result['commondata_path']}
- account mode: {result['trade_mode_label']}
- balance: {result['balance']}
- EURUSD available: {result['eurusd_available']}
""",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
