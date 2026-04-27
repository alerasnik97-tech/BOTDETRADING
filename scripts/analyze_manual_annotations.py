from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


LEDGER_PATH = Path("EURUSD_MANUAL_ANNOTATION_LEDGER.csv")
RESULTS_PATH = Path("EURUSD_MANUAL_ANNOTATION_ANALYSIS_RESULTS.csv")
EXPECTED_ROWS = 80

HUMAN_FIELDS = [
    "liquidity_source",
    "trigger_type",
    "confirmation_type",
    "operational_context",
    "entry_motive",
    "quality_rating",
    "comment",
]
ANALYSIS_FIELDS = [
    "liquidity_source",
    "trigger_type",
    "confirmation_type",
    "operational_context",
    "entry_motive",
    "quality_rating",
]
OBJECTIVE_FIELDS = [
    "liquidity_source",
    "trigger_type",
    "confirmation_type",
    "operational_context",
    "entry_motive",
]
STAGE_ORDER = ["STAGE_1_FAST_SIGNAL", "STAGE_2_CURATED"]
DISCRIMINANTS = [
    ("confirmation_type", "immediate_rejection"),
    ("liquidity_source", "london_high"),
]


def _empty_stats() -> dict[str, float | int]:
    return {
        "n": 0,
        "tp": 0,
        "be": 0,
        "sl": 0,
        "tp_rate_total": 0.0,
        "win_rate_ex_be": 0.0,
    }


def stats(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return _empty_stats()

    tp = int((frame["outcome"] == "TP").sum())
    be = int((frame["outcome"] == "BE").sum())
    sl = int((frame["outcome"] == "SL").sum())
    denom = tp + sl
    return {
        "n": int(len(frame)),
        "tp": tp,
        "be": be,
        "sl": sl,
        "tp_rate_total": tp / len(frame),
        "win_rate_ex_be": tp / denom if denom else 0.0,
    }


def add_result(rows: list[dict[str, object]], analysis_type: str, segment: str, stage: str, frame: pd.DataFrame) -> None:
    row = {"analysis_type": analysis_type, "segment": segment, "stage": stage}
    row.update(stats(frame))
    rows.append(row)


def analyze_group(rows: list[dict[str, object]], df: pd.DataFrame, field: str) -> None:
    for value, frame in df.groupby(field, dropna=False):
        label = str(value)
        add_result(rows, field, label, "TOTAL", frame)
        for stage in STAGE_ORDER:
            add_result(rows, field, label, stage, frame.loc[frame["annotation_stage"] == stage])


def discriminant_conclusion(stage1_wr: float, stage2_wr: float, total_wr: float) -> str:
    if stage2_wr >= stage1_wr - 0.10 and total_wr >= 0.55:
        return "SOSTIENE"
    if stage2_wr >= stage1_wr - 0.25 and total_wr >= 0.45:
        return "SE_DEBILITA"
    return "COLAPSA"


def stable_candidate_report(df: pd.DataFrame, baseline_wr: float) -> tuple[list[str], list[str]]:
    stable_candidates: list[str] = []
    unstable_high_signal: list[str] = []

    for field in OBJECTIVE_FIELDS:
        for value, frame in df.groupby(field):
            total = stats(frame)
            stage1 = stats(frame.loc[frame["annotation_stage"] == "STAGE_1_FAST_SIGNAL"])
            stage2 = stats(frame.loc[frame["annotation_stage"] == "STAGE_2_CURATED"])
            stage1_denom = stage1["tp"] + stage1["sl"]
            stage2_denom = stage2["tp"] + stage2["sl"]
            stable = (
                total["n"] >= 10
                and stage1_denom >= 3
                and stage2_denom >= 3
                and total["win_rate_ex_be"] >= max(0.60, baseline_wr + 0.10)
                and stage1["win_rate_ex_be"] >= baseline_wr
                and stage2["win_rate_ex_be"] >= baseline_wr
                and abs(stage1["win_rate_ex_be"] - stage2["win_rate_ex_be"]) <= 0.20
            )
            if stable:
                stable_candidates.append(
                    f"{field}={value}: total_wr={total['win_rate_ex_be']:.1%}, "
                    f"stage1_wr={stage1['win_rate_ex_be']:.1%}, stage2_wr={stage2['win_rate_ex_be']:.1%}, n={total['n']}"
                )
            elif total["n"] >= 10 and total["win_rate_ex_be"] >= 0.60:
                unstable_high_signal.append(
                    f"{field}={value}: total_wr={total['win_rate_ex_be']:.1%}, "
                    f"stage1_wr={stage1['win_rate_ex_be']:.1%}, stage2_wr={stage2['win_rate_ex_be']:.1%}, n={total['n']}"
                )

    return stable_candidates, unstable_high_signal


def determine_verdict(stable_candidates: list[str]) -> str:
    if len(stable_candidates) >= 2:
        return "FULLY_TRANSLATABLE"
    if len(stable_candidates) == 1:
        return "PARTIALLY_TRANSLATABLE"
    return "NOT_TRANSLATABLE"


def determine_decision(verdict: str, stable_candidates: list[str]) -> str:
    if verdict in {"FULLY_TRANSLATABLE", "PARTIALLY_TRANSLATABLE"} and stable_candidates:
        return "DESIGN_OBJECTIVE_HYPOTHESIS"
    return "STOP_AND_FREEZE"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the consolidated EURUSD manual annotation ledger.")
    parser.add_argument("--ledger", default=str(LEDGER_PATH), help="Ledger CSV path.")
    parser.add_argument("--results", default=str(RESULTS_PATH), help="Results CSV path.")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    results_path = Path(args.results)

    print("=" * 60)
    print("EURUSD MANUAL ANNOTATION ANALYSIS")
    print("=" * 60)
    print(f"Ledger path: {ledger_path}")

    if not ledger_path.exists():
        print(f"\n[ERROR] Ledger not found: {ledger_path}")
        return 1

    df = pd.read_csv(ledger_path)
    if len(df) != EXPECTED_ROWS:
        print(f"\n[ERROR] Expected {EXPECTED_ROWS} rows, found {len(df)}")
        return 1

    if "annotation_status" not in df.columns:
        print("\n[ERROR] annotation_status column not found. Run validation first.")
        return 1

    ready_rows = int((df["annotation_status"] == "READY_FOR_ANALYSIS").sum())
    if ready_rows != len(df):
        print(f"\n[ERROR] Ledger not ready for analysis: {ready_rows}/{len(df)} READY_FOR_ANALYSIS")
        return 1

    rows: list[dict[str, object]] = []

    add_result(rows, "overall", "TOTAL", "TOTAL", df)
    for stage in STAGE_ORDER:
        add_result(rows, "overall", stage, stage, df.loc[df["annotation_stage"] == stage])

    for field in ANALYSIS_FIELDS:
        analyze_group(rows, df, field)

    analyze_group(rows, df, "side")
    add_result(rows, "time_split", "5am-6am", "TOTAL", df.loc[df["time_block"] == "5am-6am"])
    add_result(rows, "time_split", "REST", "TOTAL", df.loc[df["time_block"] != "5am-6am"])
    for stage in STAGE_ORDER:
        stage_frame = df.loc[df["annotation_stage"] == stage]
        add_result(
            rows,
            "time_split",
            "5am-6am",
            stage,
            stage_frame.loc[stage_frame["time_block"] == "5am-6am"],
        )
        add_result(
            rows,
            "time_split",
            "REST",
            stage,
            stage_frame.loc[stage_frame["time_block"] != "5am-6am"],
        )

    combo = (
        df.groupby(["liquidity_source", "trigger_type", "confirmation_type"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    combo = combo.loc[combo["n"] >= 5].copy()
    for _, row in combo.iterrows():
        mask = (
            (df["liquidity_source"] == row["liquidity_source"])
            & (df["trigger_type"] == row["trigger_type"])
            & (df["confirmation_type"] == row["confirmation_type"])
        )
        label = f"{row['liquidity_source']} | {row['trigger_type']} | {row['confirmation_type']}"
        add_result(rows, "simple_combo", label, "TOTAL", df.loc[mask])

    discriminant_rows: list[dict[str, object]] = []
    discriminant_summary: list[str] = []
    for field, value in DISCRIMINANTS:
        stage_frames = {
            "STAGE_1_FAST_SIGNAL": df.loc[(df["annotation_stage"] == "STAGE_1_FAST_SIGNAL") & (df[field] == value)],
            "STAGE_2_CURATED": df.loc[(df["annotation_stage"] == "STAGE_2_CURATED") & (df[field] == value)],
            "TOTAL": df.loc[df[field] == value],
        }
        stage_stats = {stage: stats(frame) for stage, frame in stage_frames.items()}
        conclusion = discriminant_conclusion(
            stage_stats["STAGE_1_FAST_SIGNAL"]["win_rate_ex_be"],
            stage_stats["STAGE_2_CURATED"]["win_rate_ex_be"],
            stage_stats["TOTAL"]["win_rate_ex_be"],
        )
        for stage, stage_stat in stage_stats.items():
            discriminant_rows.append(
                {
                    "analysis_type": "stage1_discriminant",
                    "segment": f"{field}={value}",
                    "stage": stage,
                    **stage_stat,
                    "conclusion": conclusion,
                }
            )
        discriminant_summary.append(
            f"{field}={value}: stage1_wr={stage_stats['STAGE_1_FAST_SIGNAL']['win_rate_ex_be']:.1%}, "
            f"stage2_wr={stage_stats['STAGE_2_CURATED']['win_rate_ex_be']:.1%}, "
            f"total_wr={stage_stats['TOTAL']['win_rate_ex_be']:.1%}, conclusion={conclusion}"
        )

    baseline = stats(df)
    stable_candidates, unstable_high_signal = stable_candidate_report(df, baseline["win_rate_ex_be"])
    verdict = determine_verdict(stable_candidates)
    decision = determine_decision(verdict, stable_candidates)

    summary_rows = [
        {
            "analysis_type": "summary",
            "segment": "TOTAL",
            "stage": "TOTAL",
            **baseline,
            "baseline_tp_rate_total": baseline["tp_rate_total"],
            "baseline_win_rate_ex_be": baseline["win_rate_ex_be"],
            "stable_candidate_count": len(stable_candidates),
            "unstable_high_signal_count": len(unstable_high_signal),
            "verdict": verdict,
            "decision": decision,
        }
    ]

    results_df = pd.DataFrame(rows + discriminant_rows + summary_rows)
    results_df.to_csv(results_path, index=False)

    print(f"\nTotal rows analyzed: {len(df)}")
    print(f"TP/BE/SL: {baseline['tp']}/{baseline['be']}/{baseline['sl']}")
    print(f"Baseline TP rate: {baseline['tp_rate_total']:.1%}")
    print(f"Baseline win rate ex-BE: {baseline['win_rate_ex_be']:.1%}")

    print("\nStage comparison:")
    for stage in STAGE_ORDER:
        stage_stat = stats(df.loc[df["annotation_stage"] == stage])
        print(
            f"  {stage}: n={stage_stat['n']}, TP={stage_stat['tp']}, BE={stage_stat['be']}, "
            f"SL={stage_stat['sl']}, wr_ex_be={stage_stat['win_rate_ex_be']:.1%}"
        )

    print("\nStage 1 discriminant robustness:")
    for line in discriminant_summary:
        print(f"  - {line}")

    print("\nStable candidates:")
    if stable_candidates:
        for line in stable_candidates:
            print(f"  - {line}")
    else:
        print("  None")

    print("\nUnstable high-signal categories:")
    if unstable_high_signal:
        for line in unstable_high_signal:
            print(f"  - {line}")
    else:
        print("  None")

    print(f"\nFINAL VERDICT: {verdict}")
    print(f"FINAL DECISION: {decision}")
    print(f"Results saved to: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
