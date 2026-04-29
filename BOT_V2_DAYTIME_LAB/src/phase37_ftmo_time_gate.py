from __future__ import annotations

import json

from phase37_ftmo_trial_support import OUT, detect_symbol, time_gate, write_json, write_text


def write_outputs() -> dict:
    symbol_status = detect_symbol()
    status = time_gate(symbol_status)
    write_json(OUT / "time_gate" / "phase37_time_gate.json", status)
    write_text(
        OUT / "time_gate" / "phase37_time_gate.md",
        f"""
# Phase37 Time Gate

- state: {status['state']}
- NY time: {status['ny_time']}
- Argentina time: {status['argentina_time']}
- weekday NY: {status['weekday_ny']}
- server time validated: {status['server_time_validated']}
- server offset seconds: {status['server_offset_seconds']}
- reason: {status['reason']}
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
