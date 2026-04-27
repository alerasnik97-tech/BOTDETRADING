from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.forward_telemetry_lib import (
    append_trace_rows,
    build_scoreboard_trace_rows,
    load_trace_frame,
    source_hash_summary,
    telemetry_snapshot_by_line,
)
from scratch.post_hardening_drift_lib import (
    LINE_CONFIGS,
    REPORT_JSON,
    SCOREBOARD_CSV,
    compute_metrics,
    load_forward_bundle,
    load_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def build_scoreboard(report_data: dict | None = None) -> pd.DataFrame:
    report = report_data
    if report is None and REPORT_JSON.exists():
        report = load_json(REPORT_JSON)
    if report is None:
        report = {"lines": {}}

    telemetry_trace = load_trace_frame()
    telemetry_snapshot = telemetry_snapshot_by_line(telemetry_trace)
    telemetry_source = source_hash_summary()

    rows: list[dict] = []
    for line_name, config in LINE_CONFIGS.items():
        bundle = load_forward_bundle(line_name)
        metrics = compute_metrics(bundle["standardized_trades"], config["level_order"])
        report_line = report.get("lines", {}).get(line_name, {})
        line_telemetry = telemetry_snapshot.get(line_name, {})
        rows.append(
            {
                "Line": line_name,
                "Sample_N": metrics["performance_distribution"]["n"],
                "PF_Forward": metrics["performance_distribution"]["pf"],
                "Exp_Forward": metrics["performance_distribution"]["expectancy"],
                "Max_DD_R": metrics["performance_distribution"]["max_dd"],
                "Last_Activity": metrics["last_activity_ny"] or "N/A",
                "Drift_R": report_line.get("drift_r", ""),
                "Drift_Label": report_line.get("verdict", "UNKNOWN"),
                "Drift_Comparable": json.dumps(report_line.get("comparable_dimensions", {}), ensure_ascii=True),
                "Drift_Governance_Mode": report.get("tribunal_integration_mode", "TRIBUNAL_MONITOR_ONLY"),
                "Telemetry_Execution_Fidelity": line_telemetry.get("execution_fidelity", "UNAVAILABLE"),
                "Telemetry_Blocking_Fidelity": line_telemetry.get("blocking_fidelity", "UNAVAILABLE"),
                "Telemetry_Last_Guard_Status": line_telemetry.get("last_guard_status", ""),
                "Telemetry_Last_Incident": line_telemetry.get("last_incident_code", ""),
                "Telemetry_Lineage_Coverage": line_telemetry.get("lineage_coverage", 0.0),
                "Telemetry_Official_Event_Count": line_telemetry.get("official_trace_events", 0),
                "Telemetry_Trace_Path": telemetry_source.get("trace_path", ""),
            }
        )

    scoreboard = pd.DataFrame(rows)
    scoreboard.to_csv(SCOREBOARD_CSV, index=False)
    append_trace_rows(build_scoreboard_trace_rows(scoreboard))
    return scoreboard


def main() -> None:
    scoreboard = build_scoreboard()
    logging.info(f"Scoreboard saved to {SCOREBOARD_CSV}")
    print(scoreboard.to_csv(index=False))


if __name__ == "__main__":
    main()
