from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.build_scbi_dual_line_scoreboard import build_scoreboard
from scratch.post_hardening_drift_lib import REPORT_JSON, TRIBUNAL_JSON, load_json
from scratch.run_forward_evidence_tribunal import run_tribunal


def run_reconciliation() -> dict:
    report = load_json(REPORT_JSON) if REPORT_JSON.exists() else {"lines": {}, "tribunal_integration_mode": "TRIBUNAL_MONITOR_ONLY"}
    scoreboard = build_scoreboard(report)
    tribunal = run_tribunal(report, scoreboard)
    return {
        "scoreboard_rows": int(len(scoreboard)),
        "tribunal_path": str(TRIBUNAL_JSON),
        "drift_integration_mode": tribunal["drift_integration_mode"],
        "verdicts": tribunal["verdicts"],
    }


if __name__ == "__main__":
    print(json.dumps(run_reconciliation(), indent=2))
