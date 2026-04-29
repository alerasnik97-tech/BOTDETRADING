from __future__ import annotations

import json

from phase37_ftmo_trial_support import OUT, detect_symbol, write_json, write_text


def write_outputs() -> dict:
    status = detect_symbol()
    write_json(OUT / "symbol_data_gate" / "phase37_symbol_data_gate.json", status)
    write_text(
        OUT / "symbol_data_gate" / "phase37_symbol_data_gate.md",
        f"""
# Phase37 Symbol/Data Gate

- state: {status['state']}
- symbol: {status['symbol']}
- bid: {status['bid']}
- ask: {status['ask']}
- spread_pips: {status['spread_pips']}
- min_lot: {status['min_lot']}
- lot_step: {status['lot_step']}
- stops_level: {status['stops_level']}
- freeze_level: {status['freeze_level']}
- M3 bars: {status['m3_bars']}
- H1 bars: {status['h1_bars']}
- reason: {status['reason']}
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
