from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


LEDGER_PATH = Path("EURUSD_MANUAL_ANNOTATION_LEDGER.csv")
EXPECTED_ROWS = 80

LIQUIDITY_SOURCE_VALID = [
    "previous_day_high",
    "previous_day_low",
    "asia_high",
    "asia_low",
    "london_high",
    "london_low",
    "none_unclear",
]
TRIGGER_TYPE_VALID = [
    "sweep_reclaim",
    "sweep_displacement",
    "continuation_after_break",
    "reversal_after_sweep",
    "breakout_from_compression",
    "none_unclear",
]
CONFIRMATION_TYPE_VALID = [
    "close_back_inside",
    "strong_displacement_bar",
    "structure_break",
    "reclaim_then_go",
    "immediate_rejection",
    "none_unclear",
]
OPERATIONAL_CONTEXT_VALID = [
    "london_open_drive",
    "london_continuation",
    "london_reversal",
    "pre_ny_transition",
    "early_ny_followthrough",
    "none_unclear",
]
ENTRY_MOTIVE_VALID = [
    "liquidity",
    "displacement",
    "reclaim",
    "time_window",
    "confluence",
    "none_unclear",
]
QUALITY_RATING_VALID = ["A", "B", "C"]

TAXONOMY = {
    "liquidity_source": LIQUIDITY_SOURCE_VALID,
    "trigger_type": TRIGGER_TYPE_VALID,
    "confirmation_type": CONFIRMATION_TYPE_VALID,
    "operational_context": OPERATIONAL_CONTEXT_VALID,
    "entry_motive": ENTRY_MOTIVE_VALID,
    "quality_rating": QUALITY_RATING_VALID,
}

HUMAN_FIELDS = list(TAXONOMY.keys()) + ["comment"]
REQUIRED_COLUMNS = [
    "trade_id",
    "outcome",
    "side",
    "time_block",
    "annotation_stage",
    "priority_tier",
    *HUMAN_FIELDS,
]
EXPECTED_STAGE_COUNTS = {
    "STAGE_1_FAST_SIGNAL": 25,
    "STAGE_2_CURATED": 55,
}
EXPECTED_STAGE_BY_PRIORITY = {
    "FAST_SIGNAL": "STAGE_1_FAST_SIGNAL",
    "FULL_SAMPLE": "STAGE_2_CURATED",
}


def _string_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].fillna("").astype(str)


def _blank_mask(df: pd.DataFrame, column: str) -> pd.Series:
    return _string_series(df, column).str.strip() == ""


def _preview_trade_ids(trade_ids: list[str], limit: int = 10) -> str:
    preview = trade_ids[:limit]
    suffix = "" if len(trade_ids) <= limit else f" ... (+{len(trade_ids) - limit} more)"
    return ", ".join(preview) + suffix


def validate_ledger(ledger_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")

    df = pd.read_csv(ledger_path)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))
        return df, errors, warnings

    if len(df) != EXPECTED_ROWS:
        errors.append(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    if df["trade_id"].astype(str).duplicated().any():
        duplicate_ids = (
            df.loc[df["trade_id"].astype(str).duplicated(keep=False), "trade_id"]
            .astype(str)
            .tolist()
        )
        errors.append("Duplicate trade_id values: " + _preview_trade_ids(duplicate_ids))

    stage_counts = df["annotation_stage"].fillna("").value_counts().to_dict()
    for stage, expected_count in EXPECTED_STAGE_COUNTS.items():
        actual_count = int(stage_counts.get(stage, 0))
        if actual_count != expected_count:
            errors.append(f"annotation_stage={stage} expected {expected_count} rows, found {actual_count}")

    for priority_tier, expected_stage in EXPECTED_STAGE_BY_PRIORITY.items():
        mask = df["priority_tier"].fillna("") == priority_tier
        mismatch = mask & (df["annotation_stage"].fillna("") != expected_stage)
        if mismatch.any():
            trade_ids = df.loc[mismatch, "trade_id"].astype(str).tolist()
            errors.append(
                f"priority_tier={priority_tier} mismatched annotation_stage on trade_id: "
                + _preview_trade_ids(trade_ids)
            )

    for field in HUMAN_FIELDS:
        blank = _blank_mask(df, field)
        if blank.any():
            trade_ids = df.loc[blank, "trade_id"].astype(str).tolist()
            errors.append(f"Missing {field} on trade_id: " + _preview_trade_ids(trade_ids))

    for field, valid_values in TAXONOMY.items():
        blank = _blank_mask(df, field)
        cleaned = _string_series(df, field).str.strip()
        invalid_mask = ~blank & ~cleaned.isin(valid_values)
        if invalid_mask.any():
            invalid_rows = df.loc[invalid_mask, ["trade_id", field]].copy()
            invalid_rows[field] = cleaned.loc[invalid_mask]
            pairs = [f"{row.trade_id}={row[field]}" for _, row in invalid_rows.iterrows()]
            errors.append(f"Invalid values in {field}: " + _preview_trade_ids(pairs))

    comment_lengths = _string_series(df, "comment").str.len()
    long_comments = df.loc[comment_lengths > 200, "trade_id"].astype(str).tolist()
    if long_comments:
        warnings.append("Comments longer than 200 chars on trade_id: " + _preview_trade_ids(long_comments))

    missing_counts = pd.DataFrame({_field: _blank_mask(df, _field) for _field in HUMAN_FIELDS}).sum(axis=1)
    df["missing_human_fields_count"] = missing_counts.astype(int)
    df["annotation_status"] = "PENDING"
    df.loc[df["missing_human_fields_count"] == 0, "annotation_status"] = "READY_FOR_ANALYSIS"

    df.to_csv(ledger_path, index=False)
    return df, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the consolidated EURUSD manual annotation ledger.")
    parser.add_argument("--ledger", default=str(LEDGER_PATH), help="Ledger CSV path.")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)

    print("=" * 60)
    print("MANUAL ANNOTATION VALIDATION")
    print("=" * 60)
    print(f"Ledger path: {ledger_path}")

    try:
        df, errors, warnings = validate_ledger(ledger_path)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("\nSTATUS: NOT_READY_FOR_ANALYSIS")
        return 1

    print(f"Total rows: {len(df)}")
    ready_rows = int((df["annotation_status"] == "READY_FOR_ANALYSIS").sum())
    print(f"Ready rows: {ready_rows}/{len(df)}")

    stage_summary = df["annotation_stage"].fillna("").value_counts().to_dict()
    print("Stage counts:", stage_summary)

    print("\nERRORS:")
    if errors:
        for error in errors:
            print(f"  [ERROR] {error}")
    else:
        print("  None")

    print("\nWARNINGS:")
    if warnings:
        for warning in warnings:
            print(f"  [WARNING] {warning}")
    else:
        print("  None")

    if errors or ready_rows < len(df):
        print("\n" + "=" * 60)
        print("STATUS: NOT_READY_FOR_ANALYSIS")
        print("=" * 60)
        return 1

    print("\n" + "=" * 60)
    print("STATUS: READY_FOR_ANALYSIS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
