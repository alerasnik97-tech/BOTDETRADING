from __future__ import annotations

import json

from phase37_ftmo_trial_support import OUT, account_gate, detect_symbol, lot_gate_10k, write_csv, write_json, write_text


def write_outputs() -> dict:
    account = account_gate()
    symbol = detect_symbol()
    status = lot_gate_10k(symbol, account)
    fields = [
        "balance",
        "risk",
        "stop_pips",
        "risk_usd",
        "raw_lot",
        "rounded_lot",
        "actual_risk_usd",
        "actual_risk_pct",
        "allowed",
        "reason",
    ]
    write_csv(OUT / "lot_gate" / "phase37_ftmo_lot_scenarios.csv", status["rows"], fields)
    write_json(OUT / "lot_gate" / "phase37_ftmo_lot_gate.json", status)
    write_text(
        OUT / "lot_gate" / "phase37_ftmo_lot_gate.md",
        f"""
# Phase37 FTMO Lot Gate

- state: {status['state']}
- balance: {status['balance']}
- symbol: {status['symbol']}
- min lot: {status['min_lot']}
- lot step: {status['lot_step']}
- 0.50 allowed: {status['risk_050_allowed']}
- 0.75 trial allowed: {status['risk_075_trial_allowed']}
- 1.00 allowed: {status['risk_100_allowed']}
- reason: {status['reason']}

1.00% remains prohibited. 0.75% is trial stress only.
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
