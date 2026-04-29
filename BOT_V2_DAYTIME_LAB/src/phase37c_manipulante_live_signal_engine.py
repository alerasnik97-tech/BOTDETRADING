from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_support import MANIPULANTE, OUT, detect_symbol, strategy_config_gate, time_gate, write_csv, write_json, write_text


PHASE_OUT = OUT.parent / "phase37c_full_auto_ftmo_trial_bootstrap"


def evaluate_live_signal(*, news_gate: str = "NO_TRADE", data_gate: str = "NO_TRADE") -> dict[str, Any]:
    config = strategy_config_gate()
    symbol = detect_symbol()
    session = time_gate(symbol)
    equivalence_rows = [
        {"component": "authority", "expected": "PHASE25_AUTHORITY", "evidence": config.get("state"), "status": "PASS" if config.get("state") == "MANIPULANTE_MATCH" else "FAIL"},
        {"component": "symbol", "expected": "EURUSD", "evidence": symbol.get("symbol"), "status": "PASS" if symbol.get("symbol") == "EURUSD" else "FAIL"},
        {"component": "H1 Fractal Sweep", "expected": "Phase25 exact implementation", "evidence": "not mapped to callable live code", "status": "FAIL"},
        {"component": "First M3 CHOCH", "expected": "Phase25 exact implementation", "evidence": "not mapped to callable live code", "status": "FAIL"},
        {"component": "BF 70", "expected": "0.7 body filter", "evidence": "config only", "status": "REVIEW"},
        {"component": "TP", "expected": "1.4R", "evidence": "config", "status": "PASS" if config.get("state") == "MANIPULANTE_MATCH" else "FAIL"},
        {"component": "BE", "expected": "0.4R", "evidence": "config", "status": "PASS" if config.get("state") == "MANIPULANTE_MATCH" else "FAIL"},
        {"component": "News/Data gates", "expected": "ALLOW", "evidence": f"news={news_gate}; data={data_gate}", "status": "PASS" if news_gate == "ALLOW" and data_gate == "ALLOW" else "FAIL"},
    ]
    # The engine is intentionally fail-closed until the exact Phase25 sweep and
    # CHOCH code is mapped into a live callable contract.
    if config.get("state") != "MANIPULANTE_MATCH":
        state = "SIGNAL_ENGINE_REQUIRES_REPAIR"
        signal_status = "ERROR_FAIL_CLOSED"
        reason = "MANIPULANTE config gate does not match"
    elif news_gate != "ALLOW" or data_gate != "ALLOW" or session.get("state") != "ALLOW":
        state = "SIGNAL_ENGINE_REQUIRES_REPAIR"
        signal_status = "NO_TRADE_GATE_BLOCK"
        reason = "Required gates are not ALLOW"
    else:
        state = "SIGNAL_ENGINE_REQUIRES_REPAIR"
        signal_status = "ERROR_FAIL_CLOSED"
        reason = "Exact Phase25 H1 sweep / First M3 CHOCH equivalence has not been proven"
    payload = {
        "timestamp_utc": symbol.get("timestamp_utc"),
        "state": state,
        "signal_status": signal_status,
        "signal_ready": False,
        "config_gate": config,
        "symbol_gate": symbol,
        "time_gate": session,
        "equivalence_rows": equivalence_rows,
        "reason": reason,
    }
    return payload


def write_outputs(news_gate: str = "NO_TRADE", data_gate: str = "NO_TRADE") -> dict[str, Any]:
    status = evaluate_live_signal(news_gate=news_gate, data_gate=data_gate)
    write_csv(PHASE_OUT / "signal_engine" / "phase37c_signal_equivalence_check.csv", status["equivalence_rows"], ["component", "expected", "evidence", "status"])
    write_json(PHASE_OUT / "signal_engine" / "phase37c_signal_engine.json", status)
    write_text(
        PHASE_OUT / "signal_engine" / "phase37c_signal_engine.md",
        f"""
# Phase37C MANIPULANTE Live Signal Engine

- state: {status['state']}
- signal_status: {status['signal_status']}
- signal_ready: {status['signal_ready']}
- reason: {status['reason']}

This module does not invent sweep/CHOCH definitions. It remains fail-closed
until exact Phase25 live equivalence is proven.
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
