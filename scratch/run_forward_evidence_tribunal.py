from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.forward_telemetry_lib import append_trace_rows, build_tribunal_trace_rows, source_hash_summary
from scratch.post_hardening_drift_lib import REPORT_JSON, SCOREBOARD_CSV, TRIBUNAL_JSON, load_json, now_utc_iso, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def judge_line(row: pd.Series, *, drift_mode: str, drift_label: str, drift_comparable: dict[str, bool]) -> tuple[str, bool]:
    n = int(row["Sample_N"])
    pf = float(row["PF_Forward"])
    dd = float(row["Max_DD_R"])

    if dd <= -6.0:
        verdict = "SUSPENDED (DD Breach)"
    elif n < 10:
        verdict = "PAPER_ONLY (Gathering Sample)"
    elif n < 20:
        verdict = "PROMOTION_BLOCKED (Early Failure)" if pf < 1.0 else "PAPER_ONLY (Gathering Sample)"
    elif n < 40:
        if pf > 2.2 and dd > -5.0:
            verdict = "DEMO_ELIGIBLE"
        elif 1.5 <= pf <= 2.2:
            verdict = "FOLLOW_ON_OBSERVATION_REQUIRED"
        elif pf < 1.0:
            verdict = "SUSPENDED (Negative Expectancy)"
        else:
            verdict = "PAPER_ONLY"
    else:
        verdict = "REAL_ELIGIBLE (Subject to Demo Validation)" if pf > 1.8 else "FOLLOW_ON_OBSERVATION_REQUIRED"

    drift_gate_applied = False
    if (
        drift_mode == "TRIBUNAL_SAFE_TO_USE"
        and drift_label in {"STRUCTURAL_DRIFT", "DATA_OR_PIPELINE_DRIFT"}
        and any(drift_comparable.values())
    ):
        drift_gate_applied = True
        verdict = "PROMOTION_BLOCKED (Drift Gate)" if n >= 20 else "FOLLOW_ON_OBSERVATION_REQUIRED (Drift Gate)"

    return verdict, drift_gate_applied


def run_tribunal(report_data: dict | None = None, scoreboard_df: pd.DataFrame | None = None) -> dict:
    report = report_data
    if report is None:
        report = load_json(REPORT_JSON) if REPORT_JSON.exists() else {"lines": {}, "tribunal_integration_mode": "TRIBUNAL_MONITOR_ONLY"}

    if scoreboard_df is None:
        scoreboard_df = pd.read_csv(SCOREBOARD_CSV)

    telemetry_source = source_hash_summary()
    verdicts: list[dict] = []
    for _, row in scoreboard_df.iterrows():
        line = row["Line"]
        report_line = report.get("lines", {}).get(line, {})
        verdict, drift_gate_applied = judge_line(
            row,
            drift_mode=report.get("tribunal_integration_mode", "TRIBUNAL_MONITOR_ONLY"),
            drift_label=report_line.get("verdict", "UNKNOWN"),
            drift_comparable=report_line.get("comparable_dimensions", {}),
        )
        verdicts.append(
            {
                "line": line,
                "verdict": verdict,
                "n": int(row["Sample_N"]),
                "pf": float(row["PF_Forward"]),
                "dd": float(row["Max_DD_R"]),
                "drift_label": report_line.get("verdict", "UNKNOWN"),
                "drift_r": report_line.get("drift_r"),
                "drift_gate_applied": drift_gate_applied,
                "telemetry_execution_fidelity": row.get("Telemetry_Execution_Fidelity", "UNAVAILABLE"),
                "telemetry_blocking_fidelity": row.get("Telemetry_Blocking_Fidelity", "UNAVAILABLE"),
                "telemetry_last_guard_status": row.get("Telemetry_Last_Guard_Status", ""),
                "telemetry_last_incident": row.get("Telemetry_Last_Incident", ""),
                "telemetry_lineage_coverage": float(row.get("Telemetry_Lineage_Coverage", 0.0)),
            }
        )
        logging.info(f"Verdict for {line}: {verdict}")

    summary = {
        "timestamp": now_utc_iso(),
        "drift_integration_mode": report.get("tribunal_integration_mode", "TRIBUNAL_MONITOR_ONLY"),
        "drift_validation_status": report.get("monitor_validation_verdict", "UNKNOWN"),
        "telemetry_trace_path": telemetry_source.get("trace_path", ""),
        "verdicts": verdicts,
    }
    write_json(TRIBUNAL_JSON, summary)
    append_trace_rows(build_tribunal_trace_rows(summary))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_tribunal(), indent=2))
