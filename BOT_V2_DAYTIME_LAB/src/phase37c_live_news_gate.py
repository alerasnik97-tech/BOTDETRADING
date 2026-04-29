from __future__ import annotations

import json

from phase37_ftmo_live_news_consumer import write_outputs as write_phase37_news_outputs
from phase37_ftmo_trial_support import OUT, write_json, write_text


PHASE_OUT = OUT.parent / "phase37c_full_auto_ftmo_trial_bootstrap"


def write_outputs() -> dict:
    status = write_phase37_news_outputs()
    write_json(PHASE_OUT / "live_news_gate" / "phase37c_live_news_gate.json", status)
    write_text(
        PHASE_OUT / "live_news_gate" / "phase37c_live_news_gate.md",
        f"""
# Phase37C Live News Gate

- today loaded: {status['today_loaded']}
- week loaded: {status['week_loaded']}
- source: {status['source']}
- state: {status['state']}
- gate: {status['gate']}
- next blocking event: {status['next_blocking_event']}
- blocking event: {status['blocking_event']}
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
