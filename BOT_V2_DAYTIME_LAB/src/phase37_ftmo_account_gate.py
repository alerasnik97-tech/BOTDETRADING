from __future__ import annotations

import json

from phase37_ftmo_trial_support import OUT, account_gate, write_json, write_text


def write_outputs() -> dict:
    status = account_gate()
    write_json(OUT / "account_gate" / "phase37_ftmo_account_gate.json", status)
    write_text(
        OUT / "account_gate" / "phase37_ftmo_account_gate.md",
        f"""
# Phase37 FTMO Account Gate

- state: {status['state']}
- FTMO demo/trial confirmed: {status['ftmo_demo_trial_confirmed']}
- company: {status['company']}
- server: {status['server']}
- account mode: {status['trade_mode_label']}
- balance: {status['balance']}
- currency: {status['currency']}
- terminal trade allowed: {status['terminal_trade_allowed']}
- reason: {status['reason']}

Login is masked and no credentials are stored.
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
