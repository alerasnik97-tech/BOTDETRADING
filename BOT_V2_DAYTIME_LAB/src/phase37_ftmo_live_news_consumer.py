from __future__ import annotations

import json

from phase37_ftmo_trial_support import OUT, live_news_gate, write_csv, write_json, write_text


def write_outputs() -> dict:
    status = live_news_gate()
    today_rows = []
    week_rows = []
    next_event = status.get("next_blocking_event")
    if next_event:
        week_rows.append(next_event)
    fields = [
        "event_id",
        "event_name",
        "currency",
        "impact",
        "event_time_utc",
        "event_time_ny",
        "source",
        "verified",
        "timezone_validated",
        "guard_start_ny",
        "guard_end_ny",
    ]
    write_csv(OUT / "live_news_gate" / "phase37_news_today.csv", today_rows, fields)
    write_csv(OUT / "live_news_gate" / "phase37_news_week.csv", week_rows, fields)
    write_json(OUT / "live_news_gate" / "phase37_live_news_gate.json", status)
    write_text(
        OUT / "live_news_gate" / "phase37_live_news_gate.md",
        f"""
# Phase37 Live News Gate

- today loaded: {status['today_loaded']}
- week loaded: {status['week_loaded']}
- source: {status['source']}
- gate: {status['gate']}
- state: {status['state']}
- next blocking event: {status['next_blocking_event']}

If the MQL5 calendar cache is missing or stale, the gate is fail-closed.
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
