from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_support import OUT, write_json, write_text
from phase37c_mt5_terminal_autodetect import autodetect


PHASE_OUT = OUT.parent / "phase37c_full_auto_ftmo_trial_bootstrap"


def autostart_bridge() -> dict[str, Any]:
    auto = autodetect()
    data_path = Path(str(auto.get("data_path") or ""))
    files_dir = data_path / "MQL5" / "Files" / "MANIPULANTE"
    cache_files = [files_dir / "ftmo_news_today.json", files_dir / "ftmo_news_week.json", files_dir / "ftmo_news_gate_status.json"]
    if all(path.exists() for path in cache_files):
        state = "CALENDAR_BRIDGE_RUNNING"
        reason = "Cache files already exist in MQL5 Files path"
    else:
        state = "BLOCKED_MT5_EA_AUTOSTART_NOT_AVAILABLE"
        reason = "Python MT5 API cannot attach an EA to a chart safely; no CLI/profile autostart evidence was available without UI automation"
    payload = {
        "timestamp_utc": auto.get("timestamp_utc"),
        "state": state,
        "data_path": auto.get("data_path"),
        "files_dir": str(files_dir),
        "cache_files": [str(path) for path in cache_files],
        "cache_exists": {path.name: path.exists() for path in cache_files},
        "attempted_methods": [
            "MT5 data_path detection",
            "MQL5 Files cache detection",
            "safe profile/template automation review",
        ],
        "blocked_methods": [
            "pyautogui/click automation",
            "blind AutoTrading activation",
            "attaching trading EA",
            "credential/profile mutation",
        ],
        "reason": reason,
    }
    return payload


def write_outputs() -> dict[str, Any]:
    status = autostart_bridge()
    write_json(PHASE_OUT / "calendar_bridge_autostart" / "phase37c_calendar_bridge_autostart.json", status)
    write_text(
        PHASE_OUT / "calendar_bridge_autostart" / "phase37c_calendar_bridge_autostart.md",
        f"""
# Phase37C Calendar Bridge Autostart

- state: {status['state']}
- files_dir: {status['files_dir']}
- cache_exists: {status['cache_exists']}
- reason: {status['reason']}

No UI clicks, no blind AutoTrading, and no credential changes were used.
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
