from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.forward_telemetry_lib import (
    TRACE_CSV,
    build_trace_snapshot,
    load_trace_frame,
    sha256_file,
    source_hash_summary,
    telemetry_snapshot_by_line,
    write_trace_snapshot,
)
from scratch.prop_firm_risk_guards import run_all_guards
from scratch.run_post_hardening_drift_reconciliation import main as run_post_hardening_drift_stack

SCOREBOARD_CSV = ROOT / "results" / "SCBI_DUAL_LINE_SCOREBOARD.csv"
TRIBUNAL_JSON = ROOT / "results" / "SCBI_FORWARD_TRIBUNAL_SUMMARY.json"
STATUS_JSON = ROOT / "FORWARD_TELEMETRY_STATUS.json"


def load_scoreboard_metrics() -> dict[str, dict[str, Any]]:
    if not SCOREBOARD_CSV.exists():
        return {}
    frame = pd.read_csv(SCOREBOARD_CSV)
    metrics: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        metrics[str(row["Line"])] = {
            "Sample_N": int(row["Sample_N"]),
            "PF_Forward": round(float(row["PF_Forward"]), 4),
            "Exp_Forward": round(float(row["Exp_Forward"]), 4),
            "Max_DD_R": round(float(row["Max_DD_R"]), 4),
        }
    return metrics


def load_tribunal_metrics() -> dict[str, dict[str, Any]]:
    if not TRIBUNAL_JSON.exists():
        return {}
    payload = json.loads(TRIBUNAL_JSON.read_text(encoding="utf-8"))
    metrics: dict[str, dict[str, Any]] = {}
    for verdict in payload.get("verdicts", []):
        line_name = str(verdict.get("line", ""))
        metrics[line_name] = {
            "n": int(verdict.get("n", 0)),
            "pf": round(float(verdict.get("pf", 0.0)), 4),
            "dd": round(float(verdict.get("dd", 0.0)), 4),
        }
    return metrics


def compare_metrics(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"status": "PASS", "lines": {}}
    for line_name, baseline in before.items():
        current = after.get(line_name)
        line_status = "PASS" if current == baseline else "CHANGED"
        if line_status != "PASS":
            comparison["status"] = "FAIL"
        comparison["lines"][line_name] = {
            "before": baseline,
            "after": current,
            "status": line_status,
        }
    return comparison


def run_once() -> dict[str, Any]:
    guard_report = run_all_guards(emit_trace=False)
    snapshot_rows = build_trace_snapshot(guard_report, run_id=guard_report.get("run_id") or None)
    trace_snapshot = write_trace_snapshot(snapshot_rows)
    run_post_hardening_drift_stack()
    telemetry_source = source_hash_summary()
    return {
        "guard_report": guard_report,
        "trace_snapshot": trace_snapshot,
        "telemetry_source": telemetry_source,
        "trace_hash": sha256_file(TRACE_CSV) if TRACE_CSV.exists() else "",
        "trace_rows": int(len(load_trace_frame())),
    }


def final_decision(summary: dict[str, Any]) -> str:
    checks = [
        summary["idempotence_check"]["status"] == "PASS",
        summary["scoreboard_compatibility"]["status"] == "PASS",
        summary["tribunal_compatibility"]["status"] == "PASS",
        bool(summary["second_pass"]["trace_rows"] > 0),
    ]
    if all(checks):
        return "FORWARD_TELEMETRY_HARDENING_CONFIRMED"
    return "FORWARD_TELEMETRY_HARDENING_INCOMPLETE"


def main() -> None:
    scoreboard_before = load_scoreboard_metrics()
    tribunal_before = load_tribunal_metrics()

    first = run_once()
    scoreboard_after_first = load_scoreboard_metrics()
    tribunal_after_first = load_tribunal_metrics()

    second = run_once()
    scoreboard_after_second = load_scoreboard_metrics()
    tribunal_after_second = load_tribunal_metrics()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_path": first["telemetry_source"].get("trace_path", ""),
        "first_pass": {
            "trace_rows": first["trace_rows"],
            "trace_hash": first["trace_hash"],
        },
        "second_pass": {
            "trace_rows": second["trace_rows"],
            "trace_hash": second["trace_hash"],
        },
        "idempotence_check": {
            "status": "PASS" if first["trace_rows"] == second["trace_rows"] and first["trace_hash"] == second["trace_hash"] else "FAIL",
            "first_rows": first["trace_rows"],
            "second_rows": second["trace_rows"],
            "first_hash": first["trace_hash"],
            "second_hash": second["trace_hash"],
        },
        "scoreboard_compatibility": compare_metrics(scoreboard_before, scoreboard_after_second),
        "tribunal_compatibility": compare_metrics(tribunal_before, tribunal_after_second),
        "telemetry_quality": {
            "scoreboard_after_first": scoreboard_after_first,
            "tribunal_after_first": tribunal_after_first,
            "scoreboard_after_second": scoreboard_after_second,
            "tribunal_after_second": tribunal_after_second,
        },
    }
    trace_frame = load_trace_frame()
    summary["line_telemetry_snapshot"] = telemetry_snapshot_by_line(trace_frame)
    summary["decision"] = final_decision(summary)
    summary["status"] = "CONFIRMED" if summary["decision"] == "FORWARD_TELEMETRY_HARDENING_CONFIRMED" else "INCOMPLETE"
    STATUS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
