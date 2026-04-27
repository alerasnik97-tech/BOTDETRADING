from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.build_scbi_dual_line_scoreboard import build_scoreboard
from scratch.post_hardening_drift_lib import (
    BASELINE_JSON,
    POST_STATUS_JSON,
    REPORT_JSON,
    SCOREBOARD_CSV,
    TRIBUNAL_JSON,
    VALIDATION_JSON,
    audit_previous_baseline,
    build_baselines,
    final_conclusion,
    monitor_live_forward,
    validate_monitor,
    write_json,
)
from scratch.run_forward_evidence_tribunal import run_tribunal


def main() -> None:
    baselines = build_baselines()
    previous_baseline_audit = audit_previous_baseline(baselines)
    write_json(BASELINE_JSON, baselines)

    validation = validate_monitor(baselines)
    write_json(VALIDATION_JSON, validation)

    report = monitor_live_forward(baselines, validation)
    write_json(REPORT_JSON, report)

    scoreboard = build_scoreboard(report)
    tribunal = run_tribunal(report, scoreboard)

    status = {
        "generated_at_utc": baselines["generated_at_utc"],
        "baseline_version": baselines["baseline_version"],
        "taxonomy": {
            "rebaseline_requirement": previous_baseline_audit["status"],
            "rebaseline_status": "REBASELINE_CONFIRMED",
            "monitor_status": validation["overall"]["verdict"],
            "tribunal_status": report["tribunal_integration_mode"],
        },
        "previous_baseline_audit": previous_baseline_audit,
        "validation_summary": validation["overall"],
        "paths": {
            "baseline": str(BASELINE_JSON),
            "validation": str(VALIDATION_JSON),
            "report": str(REPORT_JSON),
            "scoreboard": str(SCOREBOARD_CSV),
            "tribunal": str(TRIBUNAL_JSON),
        },
        "tribunal_summary": tribunal,
    }
    status["final_conclusion"] = final_conclusion(status)
    write_json(POST_STATUS_JSON, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
