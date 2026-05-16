from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.sequential_evidence_lib import (
    DAILY_CSV,
    LINE_CONFIGS,
    SCOREBOARD_CSV,
    STATUS_JSON,
    TRACE_CSV,
    TRIBUNAL_JSON,
    build_daily_snapshot,
    build_line_status_entry,
    build_reference_model,
    load_forward_trades,
    load_validation_summary,
    load_historical_trades,
    now_utc_iso,
    read_json,
    score_trade_path,
    validate_against_official_views,
    write_json,
)


def main() -> None:
    scoreboard_df = pd.read_csv(SCOREBOARD_CSV)
    tribunal_summary = read_json(TRIBUNAL_JSON)
    tribunal_map = {entry["line"]: entry for entry in tribunal_summary["verdicts"]}
    validation_summary = load_validation_summary()

    trace_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    line_entries: dict[str, dict] = {}
    validations: dict[str, dict[str, str]] = {}

    for line_name in LINE_CONFIGS:
        history = load_historical_trades(line_name)
        model = build_reference_model(line_name, history)
        forward = load_forward_trades(line_name)
        trace_frame = score_trade_path(line_name, forward, model)
        daily_frame = build_daily_snapshot(trace_frame)
        official_validation = validate_against_official_views(
            line_name=line_name,
            current_metrics={
                "n": 0 if trace_frame.empty else int(trace_frame.iloc[-1]["cumulative_n"]),
                "expectancy": 0.0 if trace_frame.empty else float(trace_frame.iloc[-1]["cumulative_expectancy_r"]),
                "pf": 0.0 if trace_frame.empty else float(trace_frame.iloc[-1]["cumulative_pf"]),
                "max_dd": 0.0 if trace_frame.empty else float(trace_frame.iloc[-1]["cumulative_max_dd_r"]),
            },
            scoreboard_df=scoreboard_df,
            tribunal_map=tribunal_map,
        )
        validations[line_name] = official_validation
        trace_frames.append(trace_frame)
        daily_frames.append(daily_frame)
        line_entries[line_name] = build_line_status_entry(
            line_name=line_name,
            model=model,
            forward_trades=forward,
            trace_frame=trace_frame,
            official_validation=official_validation,
            validation_summary=validation_summary,
        )

    trace_output = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    daily_output = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trace_output.to_csv(TRACE_CSV, index=False)
    daily_output.to_csv(DAILY_CSV, index=False)

    status_payload = {
        "generated_at_utc": now_utc_iso(),
        "engine": "SEQUENTIAL_FORWARD_EVIDENCE_ENGINE_V1",
        "benchmark_reference": {
            "strategy": "H6_SILVER_BULLET_HYBRID",
            "role": "Benchmark conceptual congelado",
        },
        "taxonomy": {
            "states": [
                "EVIDENCE_STILL_THIN",
                "EVIDENCE_ACCUMULATING_NORMALLY",
                "EVIDENCE_TENSE_BUT_NOT_ALARMING",
                "EVIDENCE_EARLY_WARNING",
                "EVIDENCE_MATERIALLY_UNFAVORABLE",
                "SEQUENTIAL_MODEL_NOT_RELIABLE",
            ],
            "automation_posture": "MONITOR_ONLY",
        },
        "integration_posture": {
            "unified_line_status": "ANNOTATION_ONLY",
            "tribunal": "MONITOR_ONLY",
            "scoreboard": "UNCHANGED",
        },
        "source_precedence": [
            "Ledgers oficiales forward por linea",
            "Baselines historicas canonicamente validadas",
            "Scoreboard y tribunal solo para reconciliacion fail-closed",
            "Validation json como juicio de confiabilidad del modelo, no como fuente de PnL",
        ],
        "validation": {
            "status": "PASS",
            "official_view_checks": validations,
            "validator_summary_available": validation_summary is not None,
            "validator_decision": None if validation_summary is None else validation_summary.get("decision"),
            "validator_status": None if validation_summary is None else validation_summary.get("overall", {}).get("status"),
        },
        "lines": line_entries,
    }
    write_json(STATUS_JSON, status_payload)

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_status_json": str(STATUS_JSON.relative_to(ROOT)),
                "output_trace_csv": str(TRACE_CSV.relative_to(ROOT)),
                "output_daily_csv": str(DAILY_CSV.relative_to(ROOT)),
                "line_count": len(line_entries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
