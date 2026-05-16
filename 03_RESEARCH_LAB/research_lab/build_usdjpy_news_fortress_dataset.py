from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research_lab.build_am_grade_news_dataset import build_am_grade_news_dataset
from research_lab.config import NY_TZ
from research_lab.news_filter import classify_impact, normalize_event_name, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_VAULT = PROJECT_ROOT / "05_MARKET_DATA_VAULT"
DEFAULT_USD_SOURCE_FILE = DATA_VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
DEFAULT_JPY_RAW_FILE = DATA_VAULT / "data" / "forex_factory_cache.csv"
DEFAULT_OUTPUT_FILE = DATA_VAULT / "data" / "news_usdjpy_fortress_v1.csv"

REQUIRED_CLEAN_COLUMNS: list[str] = [
    "event_id",
    "event_name_normalized",
    "currency",
    "impact_level",
    "timestamp_original",
    "timezone_original",
    "timestamp_utc",
    "timestamp_ny",
    "source_name",
    "dedupe_key",
    "validation_status",
    "news_source_tier",
    "source_url",
    "notes",
]

USD_REQUIRED_FAMILIES: tuple[str, ...] = (
    "non-farm employment change",
    "unemployment rate",
    "cpi y/y",
    "cpi m/m",
    "core cpi m/m",
    "retail sales m/m",
    "core retail sales m/m",
    "unemployment claims",
    "ism manufacturing pmi",
    "ism services pmi",
    "federal funds rate",
    "fomc press conference",
)

JPY_CRITICAL_FAMILY_IMPACTS: dict[str, str] = {
    "boj policy rate": "HIGH",
    "monetary policy statement": "HIGH",
    "boj press conference": "HIGH",
    "boj outlook report": "HIGH",
}

JPY_OPTIONAL_FAMILY_IMPACTS: dict[str, str] = {
    "tokyo core cpi y/y": "MEDIUM",
    "national core cpi y/y": "MEDIUM",
}

JPY_FAMILY_IMPACTS: dict[str, str] = {
    **JPY_CRITICAL_FAMILY_IMPACTS,
    **JPY_OPTIONAL_FAMILY_IMPACTS,
}


def _window_mask(series: pd.Series, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1)
    return (series >= start_ts) & (series < end_ts)


def _ensure_usd_source() -> None:
    if DEFAULT_USD_SOURCE_FILE.exists():
        return
    build_am_grade_news_dataset(output_path=DEFAULT_USD_SOURCE_FILE)


def _load_usd_rows(start: str, end: str) -> pd.DataFrame:
    _ensure_usd_source()
    frame = pd.read_csv(DEFAULT_USD_SOURCE_FILE, dtype=str, keep_default_na=False, low_memory=False)
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
    frame["timestamp_ny_dt"] = pd.to_datetime(frame["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    approved_mask = frame["validation_status"].astype(str).str.startswith("approved")
    frame = frame.loc[
        approved_mask
        & frame["currency"].astype(str).str.upper().eq("USD")
        & _window_mask(frame["timestamp_ny_dt"], start, end)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
    frame["source_priority"] = pd.to_numeric(frame.get("source_priority", 0), errors="coerce").fillna(0).astype(int)
    frame["selection_status"] = frame.get("selection_status", "candidate")
    return frame[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]].copy()


def _load_jpy_rows(start: str, end: str) -> pd.DataFrame:
    raw = pd.read_csv(DEFAULT_JPY_RAW_FILE, dtype=str, keep_default_na=False, low_memory=False)
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])

    raw["currency"] = raw["Currency"].astype(str).str.upper()
    raw["event_name_normalized"] = raw["Event"].map(normalize_event_name)
    raw["timestamp_utc_dt"] = pd.to_datetime(raw["DateTime"], errors="coerce", utc=True)
    raw["timestamp_ny_dt"] = raw["timestamp_utc_dt"].dt.tz_convert(NY_TZ)
    raw = raw.loc[
        raw["currency"].eq("JPY")
        & raw["timestamp_ny_dt"].notna()
        & raw["event_name_normalized"].isin(JPY_FAMILY_IMPACTS)
        & _window_mask(raw["timestamp_ny_dt"], start, end)
    ].copy()
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])

    rows: list[dict[str, Any]] = []
    for item in raw.itertuples(index=False):
        event_name = str(item.event_name_normalized)
        impact_level = JPY_FAMILY_IMPACTS[event_name]
        timestamp_utc = pd.Timestamp(item.timestamp_utc_dt).tz_convert("UTC")
        timestamp_ny = pd.Timestamp(item.timestamp_ny_dt).tz_convert(NY_TZ)
        dedupe_key = stable_hash("JPY", event_name, timestamp_ny.isoformat(), impact_level)
        raw_impact = classify_impact(getattr(item, "Impact", ""))
        rows.append(
            {
                "event_id": stable_hash("usdjpy_fortress_v1", "jpy_curated_local", dedupe_key),
                "event_name_normalized": event_name,
                "currency": "JPY",
                "impact_level": impact_level,
                "timestamp_original": timestamp_utc.isoformat(),
                "timezone_original": "utc_from_local_cache",
                "timestamp_utc": timestamp_utc.isoformat(),
                "timestamp_ny": timestamp_ny.isoformat(),
                "source_name": "usdjpy_fortress_v1_jpy_curated_local",
                "dedupe_key": dedupe_key,
                "validation_status": "approved_curated_jpy_exact_timestamp",
                "news_source_tier": "pair_curated_local",
                "source_url": "local_forex_factory_cache",
                "notes": f"raw_impact={raw_impact}; family_override={impact_level}",
                "source_priority": 0,
                "selection_status": "candidate",
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["dedupe_key", "timestamp_ny", "event_name_normalized"]).reset_index(drop=True)
    duplicated = frame.duplicated(subset=["dedupe_key"], keep="first")
    frame.loc[duplicated, "selection_status"] = "rejected_lower_priority_duplicate"
    return frame[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]].copy()


def build_usdjpy_news_fortress_dataset(
    *,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    output_path: Path = DEFAULT_OUTPUT_FILE,
) -> dict[str, Any]:
    clean_output_path = Path(output_path)
    audit_output_path = clean_output_path.with_name(clean_output_path.stem + "_audit.csv")
    summary_output_path = clean_output_path.with_name(clean_output_path.stem + "_summary.json")
    clean_output_path.parent.mkdir(parents=True, exist_ok=True)

    usd_rows = _load_usd_rows(start, end)
    jpy_rows = _load_jpy_rows(start, end)
    combined = pd.concat([usd_rows, jpy_rows], ignore_index=True)

    if combined.empty:
        clean_frame = pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS)
        audit_frame = pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
    else:
        combined = combined.sort_values(["dedupe_key", "source_priority", "event_name_normalized", "timestamp_ny"]).reset_index(drop=True)
        duplicated = combined.duplicated(subset=["dedupe_key"], keep="first")
        combined.loc[duplicated, "selection_status"] = "rejected_lower_priority_duplicate"
        audit_frame = combined.copy()
        clean_frame = audit_frame.loc[~duplicated].copy()
        clean_frame = clean_frame[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]].sort_values("timestamp_ny").reset_index(drop=True)

    clean_frame.to_csv(clean_output_path, index=False)
    audit_frame.to_csv(audit_output_path, index=False)

    clean_for_validation = clean_frame.copy()
    selected_families = set(clean_for_validation.get("event_name_normalized", pd.Series(dtype=str)).astype(str).tolist())
    selected_currencies = set(clean_for_validation.get("currency", pd.Series(dtype=str)).astype(str).str.upper().tolist())
    family_coverage: dict[str, str] = {}

    for family in USD_REQUIRED_FAMILIES:
        family_coverage[family] = "USD_CANONICAL_REUSE" if family in selected_families else "MISSING"
    for family in JPY_CRITICAL_FAMILY_IMPACTS:
        family_coverage[family] = "JPY_CURATED_LOCAL_EXACT" if family in selected_families else "MISSING"
    for family in JPY_OPTIONAL_FAMILY_IMPACTS:
        family_coverage[family] = "JPY_OPTIONAL_LOCAL_CONTEXT" if family in selected_families else "MISSING_OPTIONAL"

    critical_missing = [
        family
        for family in tuple(USD_REQUIRED_FAMILIES) + tuple(JPY_CRITICAL_FAMILY_IMPACTS.keys())
        if family_coverage.get(family, "MISSING").startswith("MISSING")
    ]
    source_approved = not critical_missing and {"USD", "JPY"}.issubset(selected_currencies)
    module_verdict = "USDJPY_READY_FOR_FIRST_RESEARCH_FAMILY_DESIGN" if source_approved else "USDJPY_NOT_READY_FIX_READINESS_FIRST"

    summary = {
        "clean_dataset_path": str(clean_output_path),
        "audit_dataset_path": str(audit_output_path),
        "summary_dataset_path": str(summary_output_path),
        "raw_source_path_usd": str(DEFAULT_USD_SOURCE_FILE),
        "raw_source_path_jpy": str(DEFAULT_JPY_RAW_FILE),
        "approved_rows": int(len(clean_frame)),
        "usd_rows_reused": int((clean_frame.get("currency", pd.Series(dtype=str)).astype(str).str.upper() == "USD").sum()),
        "jpy_rows_curated": int((clean_frame.get("currency", pd.Series(dtype=str)).astype(str).str.upper() == "JPY").sum()),
        "source_approved": source_approved,
        "raw_source_name": "usdjpy_fortress_v1_local_hybrid",
        "raw_source_verdict": "USD_CANONICAL_REUSE_PLUS_JPY_LOCAL_CURATED",
        "operational_source_name": "usdjpy_fortress_v1_local_hybrid",
        "operational_source_verdict": module_verdict,
        "module_verdict": module_verdict,
        "scope_pair": "USDJPY",
        "scope_window_ny": "full_pair_specific_fortress_dataset for research, no alpha design in this cycle",
        "included_currencies": sorted(selected_currencies),
        "impact_scope": sorted(set(clean_frame.get("impact_level", pd.Series(dtype=str)).astype(str).str.upper().tolist())),
        "family_coverage": family_coverage,
        "critical_missing_families": critical_missing,
        "known_limitations": [
            "JPY speech families remain outside the hard canonical blocker unless later promoted with local audit",
            "tokyo core cpi y/y and national core cpi y/y are retained as context families, not hard-high blockers",
            "high_precision_mode is not required for first USDJPY family design and remains unavailable today",
        ],
        "high_precision_contract": {
            "required_for_first_family_design": False,
            "high_precision_ready_today": False,
            "rationale": "The first USDJPY family must start on canonical M5/M15/H1 OHLCV and normal/conservative execution. Precision can be added only if a later design proves it necessary.",
        },
    }
    summary_output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el dataset canonico USDJPY del News Fortress.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()
    summary = build_usdjpy_news_fortress_dataset(start=args.start, end=args.end, output_path=args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
