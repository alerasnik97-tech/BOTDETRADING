from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.early_forward_expectation_lib import get_line_status
from scratch.sequential_evidence_lib import (
    EARLY_OR_WORSE_STATES,
    LINE_CONFIGS,
    MAX_PREFIX_CAP,
    MIN_REFERENCE_WINDOWS,
    SCORING_VERSION_LEGACY,
    SCORING_VERSION_RECALIBRATED,
    SCORING_VERSION_REFINED,
    STATE_ORDER,
    VALIDATION_JSON,
    build_reference_model,
    load_historical_trades,
    now_utc_iso,
    score_trade_path,
    write_json,
)

VALIDATION_WINDOWS = (20, 40, 60)
CASE_HORIZON = 20
CASES_PER_FAMILY = 20
ROLLING_WINDOWS_PER_HORIZON = 250


def is_warning_or_worse(state: str) -> bool:
    return state in EARLY_OR_WORSE_STATES


def is_materially_unfavorable(state: str) -> bool:
    return state == "EVIDENCE_MATERIALLY_UNFAVORABLE"


def count_state_flips(states: list[str]) -> int:
    if not states:
        return 0
    flips = 0
    previous = states[0]
    for state in states[1:]:
        if state != previous:
            flips += 1
            previous = state
    return flips


def earliest_sequential_trigger(path: pd.DataFrame) -> int | None:
    for _, row in path.iterrows():
        if row["sequential_evidence_state"] in EARLY_OR_WORSE_STATES:
            return int(row["cumulative_n"])
    return None


def earliest_checkpoint_trigger(line_name: str, path: pd.DataFrame) -> int | None:
    for _, row in path.iterrows():
        label = get_line_status(
            line_name,
            int(row["cumulative_n"]),
            {
                "pf": row["cumulative_pf"],
                "expectancy": row["cumulative_expectancy_r"],
                "max_dd": row["cumulative_max_dd_r"],
            },
        )
        if label in {"EARLY_WARNING", "OUTSIDE_EXPECTATION_ENVELOPE"}:
            return int(row["cumulative_n"])
    return None


def select_extreme_starts(
    history: pd.DataFrame,
    *,
    block_size: int,
    case_count: int,
    mode: str,
    horizon: int = CASE_HORIZON,
) -> list[int]:
    values = history["pnl_r"].astype(float).to_numpy()
    limit = len(history) - horizon + 1
    if limit <= 0:
        return []
    scored = []
    for start in range(limit):
        block_mean = float(values[start : start + block_size].mean())
        scored.append((start, block_mean))
    reverse = mode == "best"
    scored.sort(key=lambda item: item[1], reverse=reverse)

    chosen: list[int] = []
    for start, _ in scored:
        candidate_end = start + horizon - 1
        overlaps = False
        for existing_start in chosen:
            existing_end = existing_start + horizon - 1
            if not (candidate_end < existing_start or start > existing_end):
                overlaps = True
                break
        if overlaps:
            continue
        chosen.append(start)
        if len(chosen) >= case_count:
            break
    return chosen


def evenly_spaced_starts(history: pd.DataFrame, *, case_count: int, horizon: int = CASE_HORIZON) -> list[int]:
    limit = len(history) - horizon + 1
    if limit <= 0:
        return []
    if limit <= case_count:
        return list(range(limit))
    starts = sorted({int(round(value)) for value in np.linspace(0, limit - 1, num=case_count)})
    return starts


def percentile_pool(history: pd.DataFrame, *, tail: str) -> list[float]:
    values = history["pnl_r"].astype(float).to_numpy()
    quantile = 0.05 if tail == "low" else 0.95
    cutoff = float(np.quantile(values, quantile))
    if tail == "low":
        pool = sorted(float(value) for value in values if value <= cutoff)
    else:
        pool = sorted((float(value) for value in values if value >= cutoff), reverse=True)
    return pool or [float(cutoff)]


def build_replacement_case(
    base_chunk: pd.DataFrame,
    replacement_pool: list[float],
    *,
    block_size: int,
    case_index: int,
    label: str,
) -> pd.DataFrame:
    chunk = base_chunk.copy().reset_index(drop=True)
    for idx in range(min(block_size, len(chunk))):
        replacement = replacement_pool[(case_index + idx) % len(replacement_pool)]
        chunk.loc[idx, "pnl_r"] = float(replacement)
        chunk.loc[idx, "official_id"] = f"{chunk.loc[idx, 'official_id']}_{label}_{idx}"
    return chunk


def window_validation(
    line_name: str,
    history: pd.DataFrame,
    window: int,
    model,
    *,
    scoring_version: str,
) -> dict[str, object]:
    full_window_count = len(history) - window + 1
    if full_window_count <= 0:
        starts: list[int] = []
    elif full_window_count <= ROLLING_WINDOWS_PER_HORIZON:
        starts = list(range(full_window_count))
    else:
        starts = sorted({int(round(value)) for value in np.linspace(0, full_window_count - 1, num=ROLLING_WINDOWS_PER_HORIZON)})
    counts = {
        "prefixes_n10": 0,
        "prefixes_n20": 0,
        "material_hits_n10": 0,
        "material_hits_n20": 0,
        "warning_hits_n10": 0,
        "warning_hits_n20": 0,
    }
    flips: list[int] = []
    unreliable_prefixes = 0

    for start in starts:
        end = start + window - 1
        chunk = history.iloc[start : start + window].copy()
        path = score_trade_path(line_name, chunk, model, exclude_interval=(start, end), scoring_version=scoring_version)
        states = list(path["sequential_evidence_state"].astype(str).head(20))
        flips.append(count_state_flips(states))
        unreliable_prefixes += int((path["sequential_evidence_state"] == "SEQUENTIAL_MODEL_NOT_RELIABLE").head(20).sum())

        for _, row in path.iterrows():
            n = int(row["cumulative_n"])
            state = str(row["sequential_evidence_state"])
            if n <= 10:
                counts["prefixes_n10"] += 1
                if is_materially_unfavorable(state):
                    counts["material_hits_n10"] += 1
                if is_warning_or_worse(state):
                    counts["warning_hits_n10"] += 1
            if n <= 20:
                counts["prefixes_n20"] += 1
                if is_materially_unfavorable(state):
                    counts["material_hits_n20"] += 1
                if is_warning_or_worse(state):
                    counts["warning_hits_n20"] += 1

    material_rate_n10 = counts["material_hits_n10"] / counts["prefixes_n10"] if counts["prefixes_n10"] else 0.0
    material_rate_n20 = counts["material_hits_n20"] / counts["prefixes_n20"] if counts["prefixes_n20"] else 0.0
    warning_rate_n10 = counts["warning_hits_n10"] / counts["prefixes_n10"] if counts["prefixes_n10"] else 0.0
    warning_rate_n20 = counts["warning_hits_n20"] / counts["prefixes_n20"] if counts["prefixes_n20"] else 0.0
    median_flips = float(median(flips)) if flips else 0.0
    unreliable_rate = unreliable_prefixes / counts["prefixes_n20"] if counts["prefixes_n20"] else 0.0

    return {
        "window": window,
        "windows_tested": len(starts),
        "full_window_population": full_window_count,
        "material_false_positive_rate_n10": round(material_rate_n10, 4),
        "material_false_positive_rate_n20": round(material_rate_n20, 4),
        "warning_false_positive_rate_n10": round(warning_rate_n10, 4),
        "warning_false_positive_rate_n20": round(warning_rate_n20, 4),
        "median_flips_first20": round(median_flips, 4),
        "unreliable_prefix_rate_n20": round(unreliable_rate, 4),
    }


def historical_replay_summary(line_name: str, history: pd.DataFrame, model, *, scoring_version: str) -> dict[str, object]:
    replay_horizon = min(MAX_PREFIX_CAP, len(history))
    chunk = history.iloc[:replay_horizon].copy()
    path = score_trade_path(line_name, chunk, model, scoring_version=scoring_version)
    state_counts = path["sequential_evidence_state"].astype(str).value_counts().to_dict()
    confidence_values = pd.to_numeric(path["institutional_confidence_score"], errors="coerce").dropna()
    compatibility_values = pd.to_numeric(path["cumulative_compatibility_score"], errors="coerce").dropna()
    early_confidence = pd.to_numeric(path.loc[path["cumulative_n"] < 5, "institutional_confidence_score"], errors="coerce").dropna()
    return {
        "replay_horizon": replay_horizon,
        "state_counts": state_counts,
        "first_sequential_warning_n": earliest_sequential_trigger(path),
        "min_confidence_score": None if confidence_values.empty else round(float(confidence_values.min()), 4),
        "max_confidence_score": None if confidence_values.empty else round(float(confidence_values.max()), 4),
        "median_confidence_score": None if confidence_values.empty else round(float(confidence_values.median()), 4),
        "max_confidence_score_before_n5": None if early_confidence.empty else round(float(early_confidence.max()), 4),
        "median_compatibility_score": None if compatibility_values.empty else round(float(compatibility_values.median()), 4),
    }


def ugly_start_validation(line_name: str, history: pd.DataFrame, model, *, scoring_version: str) -> dict[str, object]:
    low_pool = percentile_pool(history, tail="low")
    cases: list[dict[str, object]] = []

    for block_size in (3, 5):
        for case_index, start in enumerate(select_extreme_starts(history, block_size=block_size, case_count=CASES_PER_FAMILY, mode="worst")):
            base_chunk = history.iloc[start : start + CASE_HORIZON].copy()
            path = score_trade_path(line_name, base_chunk, model, exclude_interval=(start, start + CASE_HORIZON - 1), scoring_version=scoring_version)
            cases.append(
                {
                    "family": f"worst_block_{block_size}",
                    "case_index": case_index,
                    "sequential_trigger_n": earliest_sequential_trigger(path),
                    "checkpoint_trigger_n": earliest_checkpoint_trigger(line_name, path),
                }
            )

        for case_index, start in enumerate(evenly_spaced_starts(history, case_count=CASES_PER_FAMILY)):
            base_chunk = history.iloc[start : start + CASE_HORIZON].copy()
            replaced = build_replacement_case(
                base_chunk,
                low_pool,
                block_size=block_size,
                case_index=case_index,
                label=f"LOWP5_{block_size}",
            )
            path = score_trade_path(line_name, replaced, model, exclude_interval=(start, start + CASE_HORIZON - 1), scoring_version=scoring_version)
            cases.append(
                {
                    "family": f"p05_replace_{block_size}",
                    "case_index": case_index,
                    "sequential_trigger_n": earliest_sequential_trigger(path),
                    "checkpoint_trigger_n": earliest_checkpoint_trigger(line_name, path),
                }
            )

    successes = 0
    sequential_triggers: list[int] = []
    checkpoint_triggers: list[int] = []
    downside_hits_n5 = 0
    checkpoint_hits_n5 = 0
    for case in cases:
        seq_trigger = case["sequential_trigger_n"]
        checkpoint_trigger = case["checkpoint_trigger_n"]
        if seq_trigger is not None:
            sequential_triggers.append(int(seq_trigger))
            if int(seq_trigger) <= 5:
                downside_hits_n5 += 1
        if checkpoint_trigger is not None:
            checkpoint_triggers.append(int(checkpoint_trigger))
            if int(checkpoint_trigger) <= 5:
                checkpoint_hits_n5 += 1
        if seq_trigger is not None and (checkpoint_trigger is None or int(seq_trigger) <= int(checkpoint_trigger) - 1):
            successes += 1

    earlier_rate = successes / len(cases) if cases else 0.0
    downside_warning_hit_rate_n5 = downside_hits_n5 / len(cases) if cases else 0.0
    false_calm_rate_n5 = 1.0 - downside_warning_hit_rate_n5 if cases else 0.0
    return {
        "cases_total": len(cases),
        "sequential_earlier_than_checkpoint_rate": round(earlier_rate, 4),
        "downside_warning_hit_rate_n5": round(downside_warning_hit_rate_n5, 4),
        "false_calm_rate_n5": round(false_calm_rate_n5, 4),
        "checkpoint_warning_hit_rate_n5": round(checkpoint_hits_n5 / len(cases), 4) if cases else 0.0,
        "median_sequential_trigger_n": None if not sequential_triggers else round(float(median(sequential_triggers)), 4),
        "median_checkpoint_trigger_n": None if not checkpoint_triggers else round(float(median(checkpoint_triggers)), 4),
    }


def too_good_start_validation(line_name: str, history: pd.DataFrame, model, *, scoring_version: str) -> dict[str, object]:
    high_pool = percentile_pool(history, tail="high")
    cases: list[pd.DataFrame] = []

    for block_size in (3, 5):
        for start in select_extreme_starts(history, block_size=block_size, case_count=CASES_PER_FAMILY, mode="best"):
            base_chunk = history.iloc[start : start + CASE_HORIZON].copy()
            cases.append(score_trade_path(line_name, base_chunk, model, exclude_interval=(start, start + CASE_HORIZON - 1), scoring_version=scoring_version))

        for case_index, start in enumerate(evenly_spaced_starts(history, case_count=CASES_PER_FAMILY)):
            base_chunk = history.iloc[start : start + CASE_HORIZON].copy()
            replaced = build_replacement_case(
                base_chunk,
                high_pool,
                block_size=block_size,
                case_index=case_index,
                label=f"HIGHP95_{block_size}",
            )
            cases.append(score_trade_path(line_name, replaced, model, exclude_interval=(start, start + CASE_HORIZON - 1), scoring_version=scoring_version))

    overconfident_hits = 0
    prefix_points = 0
    for path in cases:
        early_prefix = path[path["cumulative_n"] < 5]
        prefix_points += len(early_prefix)
        overconfident_hits += int((pd.to_numeric(early_prefix["institutional_confidence_score"], errors="coerce") > 90.0).sum())

    rate = overconfident_hits / prefix_points if prefix_points else 0.0
    return {
        "cases_total": len(cases),
        "prefix_points_n_lt_5": prefix_points,
        "overconfidence_rate_before_n5": round(rate, 4),
    }


def run_line_validation(line_name: str, *, scoring_version: str) -> dict[str, object]:
    history = load_historical_trades(line_name)
    model = build_reference_model(line_name, history)
    clean_results = {
        str(window): window_validation(line_name, history, window, model, scoring_version=scoring_version) for window in VALIDATION_WINDOWS
    }
    replay = historical_replay_summary(line_name, history, model, scoring_version=scoring_version)
    ugly = ugly_start_validation(line_name, history, model, scoring_version=scoring_version)
    hot = too_good_start_validation(line_name, history, model, scoring_version=scoring_version)

    material_n10 = max(float(clean_results[str(window)]["material_false_positive_rate_n10"]) for window in VALIDATION_WINDOWS)
    material_n20 = max(float(clean_results[str(window)]["material_false_positive_rate_n20"]) for window in VALIDATION_WINDOWS)
    warning_n10 = max(float(clean_results[str(window)]["warning_false_positive_rate_n10"]) for window in VALIDATION_WINDOWS)
    warning_n20 = max(float(clean_results[str(window)]["warning_false_positive_rate_n20"]) for window in VALIDATION_WINDOWS)
    median_flips = max(float(clean_results[str(window)]["median_flips_first20"]) for window in VALIDATION_WINDOWS)
    ugly_rate = float(ugly["sequential_earlier_than_checkpoint_rate"])
    hot_rate = float(hot["overconfidence_rate_before_n5"])

    checks = {
        "material_false_positive_rate_n10": {"actual": round(material_n10, 4), "target": "<=0.01", "pass": material_n10 <= 0.01},
        "material_false_positive_rate_n20": {"actual": round(material_n20, 4), "target": "<=0.03", "pass": material_n20 <= 0.03},
        "warning_false_positive_rate_n10": {"actual": round(warning_n10, 4), "target": "<=0.05", "pass": warning_n10 <= 0.05},
        "warning_false_positive_rate_n20": {"actual": round(warning_n20, 4), "target": "<=0.10", "pass": warning_n20 <= 0.10},
        "median_flips_first20": {"actual": round(median_flips, 4), "target": "<=2", "pass": median_flips <= 2.0},
        "ugly_start_earlier_rate": {"actual": round(ugly_rate, 4), "target": ">=0.50", "pass": ugly_rate >= 0.50},
        "too_good_start_overconfidence_rate": {"actual": round(hot_rate, 4), "target": "<0.05", "pass": hot_rate < 0.05},
    }

    passed = all(item["pass"] for item in checks.values())
    return {
        "line": line_name,
        "scoring_version": scoring_version,
        "status": "PASS" if passed else "FAIL",
        "model_scope": {
            "historical_rows": int(len(history)),
            "max_prefix": model.max_prefix,
            "min_reference_windows": MIN_REFERENCE_WINDOWS,
        },
        "clean_window_validation": clean_results,
        "historical_replay": replay,
        "ugly_start_validation": ugly,
        "too_good_start_validation": hot,
        "targets": checks,
    }


def comparison_summary(legacy: dict[str, object], recalibrated: dict[str, object]) -> dict[str, object]:
    legacy_ugly = legacy["ugly_start_validation"]
    recal_ugly = recalibrated["ugly_start_validation"]
    legacy_hot = legacy["too_good_start_validation"]
    recal_hot = recalibrated["too_good_start_validation"]
    legacy_replay = legacy["historical_replay"]
    recal_replay = recalibrated["historical_replay"]
    return {
        "overconfidence_rate_before_n5_delta": round(
            float(recal_hot["overconfidence_rate_before_n5"]) - float(legacy_hot["overconfidence_rate_before_n5"]),
            4,
        ),
        "overconfidence_rate_before_n5_improvement": round(
            float(legacy_hot["overconfidence_rate_before_n5"]) - float(recal_hot["overconfidence_rate_before_n5"]),
            4,
        ),
        "downside_warning_hit_rate_n5_delta": round(
            float(recal_ugly["downside_warning_hit_rate_n5"]) - float(legacy_ugly["downside_warning_hit_rate_n5"]),
            4,
        ),
        "false_calm_rate_n5_delta": round(
            float(recal_ugly["false_calm_rate_n5"]) - float(legacy_ugly["false_calm_rate_n5"]),
            4,
        ),
        "earlier_than_checkpoint_rate_delta": round(
            float(recal_ugly["sequential_earlier_than_checkpoint_rate"]) - float(legacy_ugly["sequential_earlier_than_checkpoint_rate"]),
            4,
        ),
        "max_replay_confidence_before_n5_delta": round(
            float(recal_replay["max_confidence_score_before_n5"]) - float(legacy_replay["max_confidence_score_before_n5"]),
            4,
        ),
    }


def recalibration_taxonomy(legacy: dict[str, object], recalibrated: dict[str, object]) -> dict[str, str]:
    legacy_hot = float(legacy["too_good_start_validation"]["overconfidence_rate_before_n5"])
    recal_hot = float(recalibrated["too_good_start_validation"]["overconfidence_rate_before_n5"])
    legacy_ugly = float(legacy["ugly_start_validation"]["downside_warning_hit_rate_n5"])
    recal_ugly = float(recalibrated["ugly_start_validation"]["downside_warning_hit_rate_n5"])
    legacy_false_calm = float(legacy["ugly_start_validation"]["false_calm_rate_n5"])
    recal_false_calm = float(recalibrated["ugly_start_validation"]["false_calm_rate_n5"])

    fixed = "YES" if recal_hot < 0.05 else "NO"
    improved = "YES" if recal_hot < legacy_hot else "NO"
    downside_preserved = "YES" if recal_ugly >= max(0.0, legacy_ugly - 0.05) and recal_false_calm <= min(1.0, legacy_false_calm + 0.05) else "NO"
    upside_discounted = "YES" if recal_hot < legacy_hot and recalibrated["historical_replay"]["max_confidence_score_before_n5"] < 90.0 else "NO"
    too_conservative = "YES" if recal_ugly < max(0.50, legacy_ugly - 0.15) else "NO"
    reliable = "YES" if recalibrated["status"] == "PASS" else "NO"
    return {
        "ANTI_OVERCONFIDENCE_FIXED": fixed,
        "ANTI_OVERCONFIDENCE_IMPROVED_BUT_NOT_ENOUGH": "YES" if improved == "YES" and fixed == "NO" else "NO",
        "DOWNSIDE_SENSITIVITY_PRESERVED": downside_preserved,
        "UPSIDE_EVIDENCE_PROPERLY_DISCOUNTED": upside_discounted,
        "MODEL_BECAME_TOO_CONSERVATIVE": too_conservative,
        "RECALIBRATION_NOT_RELIABLE": "NO" if reliable == "YES" else "YES",
    }


def main() -> None:
    legacy_results = {line_name: run_line_validation(line_name, scoring_version=SCORING_VERSION_LEGACY) for line_name in LINE_CONFIGS}
    v2_results = {line_name: run_line_validation(line_name, scoring_version=SCORING_VERSION_RECALIBRATED) for line_name in LINE_CONFIGS}
    refined_results = {line_name: run_line_validation(line_name, scoring_version=SCORING_VERSION_REFINED) for line_name in LINE_CONFIGS}
    
    comparisons_v2 = {
        line_name: comparison_summary(legacy_results[line_name], v2_results[line_name]) for line_name in LINE_CONFIGS
    }
    comparisons_v3 = {
        line_name: comparison_summary(v2_results[line_name], refined_results[line_name]) for line_name in LINE_CONFIGS
    }
    taxonomy = {
        line_name: recalibration_taxonomy(v2_results[line_name], refined_results[line_name]) for line_name in LINE_CONFIGS
    }
    passed = all(result["status"] == "PASS" for result in refined_results.values())
    layer_decision = "SEQUENTIAL_EVIDENCE_LAYER_CONFIRMED" if passed else "SEQUENTIAL_EVIDENCE_LAYER_NEEDS_REFINEMENT"
    recalibration_decision = "SEQUENTIAL_EVIDENCE_RECALIBRATION_CONFIRMED" if passed else "SEQUENTIAL_EVIDENCE_RECALIBRATION_FAILED"

    payload = {
        "generated_at_utc": now_utc_iso(),
        "engine": "SEQUENTIAL_FORWARD_EVIDENCE_VALIDATOR_V2",
        "decision": layer_decision,
        "recalibration_decision": recalibration_decision,
        "targets_frozen_ex_ante": {
            "material_false_positive_rate_n10": "<=0.01",
            "material_false_positive_rate_n20": "<=0.03",
            "warning_false_positive_rate_n10": "<=0.05",
            "warning_false_positive_rate_n20": "<=0.10",
            "median_flips_first20": "<=2",
            "ugly_start_earlier_rate": ">=0.50",
            "too_good_start_overconfidence_rate_before_n5": "<0.05",
            "downside_warning_hit_rate_n5_preservation": "recalibrated >= legacy - 0.05",
            "false_calm_rate_n5_preservation": "recalibrated <= legacy + 0.05",
        },
        "lines": {
            line_name: {
                "legacy": legacy_results[line_name],
                "v2": v2_results[line_name],
                "refined_v3": refined_results[line_name],
                "comparison_v2": comparisons_v2[line_name],
                "comparison_v3": comparisons_v3[line_name],
                "taxonomy": taxonomy[line_name],
            }
            for line_name in LINE_CONFIGS
        },
        "overall": {
            "status": "PASS" if passed else "FAIL",
            "decision": layer_decision,
            "recalibration_decision": recalibration_decision,
        },
    }
    write_json(VALIDATION_JSON, payload)
    print(
        json.dumps(
            {
                "status": payload["overall"]["status"],
                "decision": layer_decision,
                "recalibration_decision": recalibration_decision,
                "output": str(VALIDATION_JSON.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
