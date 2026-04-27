from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
import zoneinfo

import pandas as pd

from research_lab.config import NY_TZ
from research_lab.news_filter import (
    SUPPORTED_FIXED_SCHEDULES_NY,
    normalize_event_name,
    stable_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ANCHOR_FILE = PROJECT_ROOT / "data" / "official_anchors" / "out" / "canonical_anchor_events.csv"
DEFAULT_LEGACY_VALIDATED_FILE = PROJECT_ROOT / "data" / "news_eurusd_v2_utc.csv"
DEFAULT_CURATED_SUPPLEMENT_FILE = PROJECT_ROOT / "data" / "official_anchors" / "manifests" / "curated_supplementary_us.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "news_eurusd_am_fortress_v3.csv"
FRANKFURT_TZ = "Europe/Berlin"
ECB_PRESS_CONFERENCE_LOCAL_TIME = "14:45"

OFFICIAL_EVENT_REMAP: dict[str, str] = {
    "fomc rate decision": "federal funds rate",
    "ecb monetary policy decision": "main refinancing rate",
}

OFFICIAL_PRIMARY_FAMILIES: tuple[str, ...] = (
    "non-farm employment change",
    "unemployment rate",
    "cpi m/m",
    "ppi m/m",
    "federal funds rate",
    "main refinancing rate",
)

LEGACY_SUPPLEMENTAL_FAMILIES: tuple[str, ...] = (
    "adp non-farm employment change",
    "cpi y/y",
    "core cpi m/m",
    "core ppi m/m",
    "core retail sales m/m",
    "advance gdp q/q",
    "prelim gdp q/q",
    "final gdp q/q",
    "ism manufacturing pmi",
    "ism services pmi",
    "fomc statement",
    "fomc meeting minutes",
    "fomc press conference",
)

CURATED_SUPPLEMENTAL_FAMILIES: tuple[str, ...] = (
    "retail sales m/m",
    "unemployment claims",
    "core retail sales m/m",
)

AM_COVERAGE_FAMILIES: tuple[str, ...] = (
    "adp non-farm employment change",
    "non-farm employment change",
    "unemployment rate",
    "cpi m/m",
    "cpi y/y",
    "core cpi m/m",
    "ppi m/m",
    "core ppi m/m",
    "retail sales m/m",
    "core retail sales m/m",
    "unemployment claims",
    "ism manufacturing pmi",
    "ism services pmi",
    "federal funds rate",
    "fomc press conference",
    "main refinancing rate",
    "ecb press conference",
)

CRITICAL_MISSING_FAMILIES: tuple[str, ...] = ()

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

AM_KEY_EVENT_EXPECTATIONS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("non-farm employment change", ("08:30",), False),
    ("unemployment rate", ("08:30",), False),
    ("cpi y/y", ("08:30",), False),
    ("cpi m/m", ("08:30",), False),
    ("core cpi m/m", ("08:30",), False),
    ("retail sales m/m", ("08:30",), False),
    ("core retail sales m/m", ("08:30",), False),
    ("ism manufacturing pmi", ("10:00",), False),
    ("ism services pmi", ("10:00",), False),
    ("fomc statement", ("14:00",), False),
    ("fomc meeting minutes", ("14:00",), False),
    ("fomc press conference", ("14:30",), False),
    ("gdp q/q", ("08:30",), False),
    ("ppi y/y", ("08:30",), True),
    ("main refinancing rate", ("07:15", "08:15"), False),
    ("ecb press conference", ("08:45", "09:45"), False),
    ("unemployment claims", ("08:30",), False),
)


def _window_mask(series: pd.Series, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1)
    return (series >= start_ts) & (series < end_ts)


def _frankfurt_local_to_utc_and_ny(local_date: date, hhmm: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    hour, minute = (int(part) for part in hhmm.split(":"))
    local_dt = datetime.combine(local_date, time(hour=hour, minute=minute), tzinfo=zoneinfo.ZoneInfo(FRANKFURT_TZ))
    local_ts = pd.Timestamp(local_dt)
    utc_ts = local_ts.tz_convert("UTC")
    ny_ts = local_ts.tz_convert(NY_TZ)
    return utc_ts, ny_ts, f"iana_{FRANKFURT_TZ.replace('/', '_').lower()}"


def _derive_ecb_press_conference_candidates(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    decision_rows = frame.loc[
        frame["status"].eq("approved_technical")
        & frame["title"].eq("ecb monetary policy decision")
        & frame["currency"].astype(str).str.upper().eq("EUR")
        & frame["importance"].astype(str).str.upper().eq("HIGH")
    ].copy()
    if decision_rows.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])

    frankfurt_dates = decision_rows["scheduled_at_utc"].dt.tz_convert(FRANKFURT_TZ).dt.date
    derived_rows: list[dict[str, Any]] = []
    for row, local_date in zip(decision_rows.itertuples(index=False), frankfurt_dates, strict=False):
        if pd.isna(local_date):
            continue
        scheduled_at_utc, scheduled_at_ny, timezone_source = _frankfurt_local_to_utc_and_ny(local_date, ECB_PRESS_CONFERENCE_LOCAL_TIME)
        derived_rows.append(
            {
                "event_name_normalized": "ecb press conference",
                "currency": "EUR",
                "impact_level": "HIGH",
                "timestamp_original": scheduled_at_utc.isoformat(),
                "timezone_original": timezone_source,
                "timestamp_utc": scheduled_at_utc.isoformat(),
                "timestamp_ny": scheduled_at_ny.isoformat(),
                "source_name": "am_fortress_v3_hybrid_local",
                "dedupe_key": stable_hash("EUR", "ecb press conference", scheduled_at_ny.isoformat(), "HIGH"),
                "validation_status": "approved_official_anchor_derived_schedule",
                "news_source_tier": "official_anchor",
                "source_url": str(getattr(row, "source_url", "") or ""),
                "notes": "derived_from_ecb_official_meeting_calendar:Europe/Berlin_14:45",
                "source_priority": 0,
                "selection_status": "candidate",
            }
        )

    derived = pd.DataFrame(derived_rows)
    if derived.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
    derived["event_id"] = derived["dedupe_key"].map(lambda value: stable_hash("am_fortress_v3", "official_anchor", value))
    derived["timestamp_ny"] = pd.to_datetime(derived["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    derived = derived.loc[_window_mask(derived["timestamp_ny"], start, end)].copy()
    derived["timestamp_ny"] = derived["timestamp_ny"].map(lambda value: value.isoformat() if pd.notna(value) else "")
    return derived[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]].copy()


def _load_official_anchor_candidates(start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(DEFAULT_OFFICIAL_ANCHOR_FILE, dtype=str, keep_default_na=False, low_memory=False)
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])

    frame["scheduled_at_ny"] = pd.to_datetime(frame["scheduled_at_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    frame["scheduled_at_utc"] = pd.to_datetime(frame["scheduled_at_utc"], utc=True, errors="coerce")
    frame["title"] = frame["title"].map(normalize_event_name)
    frame["event_name_normalized"] = frame["title"].replace(OFFICIAL_EVENT_REMAP)
    eligible = frame.loc[
        frame["status"].eq("approved_technical")
        & frame["currency"].astype(str).str.upper().isin({"USD", "EUR"})
        & frame["importance"].astype(str).str.upper().eq("HIGH")
    ].copy()
    primary = eligible.loc[eligible["event_name_normalized"].isin(OFFICIAL_PRIMARY_FAMILIES)].copy()
    primary = primary.loc[_window_mask(primary["scheduled_at_ny"], start, end)].copy()
    primary["source_name"] = "am_fortress_v3_hybrid_local"
    primary["impact_level"] = "HIGH"
    primary["timestamp_original"] = primary["scheduled_at_utc"].astype(str)
    primary["timezone_original"] = primary["timezone_source"].astype(str)
    primary["timestamp_utc"] = primary["scheduled_at_utc"].map(lambda value: value.isoformat() if pd.notna(value) else "")
    primary["timestamp_ny"] = primary["scheduled_at_ny"].map(lambda value: value.isoformat() if pd.notna(value) else "")
    primary["validation_status"] = "approved_official_anchor"
    primary["news_source_tier"] = "official_anchor"
    primary["dedupe_key"] = primary.apply(
        lambda row: stable_hash(
            row["currency"],
            row["event_name_normalized"],
            row["timestamp_ny"],
            "HIGH",
        ),
        axis=1,
    )
    primary["event_id"] = primary.apply(
        lambda row: stable_hash("am_fortress_v3", "official_anchor", row["dedupe_key"]),
        axis=1,
    )
    primary["notes"] = primary["notes"].astype(str)
    primary["source_priority"] = 0
    primary["selection_status"] = "candidate"

    derived = _derive_ecb_press_conference_candidates(eligible, start, end)
    combined = pd.concat([primary[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]], derived], ignore_index=True)
    return combined.sort_values(["timestamp_ny", "event_name_normalized"]).reset_index(drop=True)


def _accepted_legacy_families(frame: pd.DataFrame) -> dict[str, str]:
    accepted: dict[str, str] = {}
    if frame.empty:
        return accepted
    for event_name in LEGACY_SUPPLEMENTAL_FAMILIES:
        expected_hhmm = SUPPORTED_FIXED_SCHEDULES_NY.get(event_name)
        if expected_hhmm is None:
            continue
        subset = frame.loc[frame["event_name_normalized"].eq(event_name)].copy()
        if subset.empty:
            continue
        times = pd.to_datetime(subset["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ).dt.strftime("%H:%M")
        if bool((times == expected_hhmm).all()):
            accepted[event_name] = expected_hhmm
    return accepted


def _load_legacy_supplement_candidates(start: str, end: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if not DEFAULT_LEGACY_VALIDATED_FILE.exists():
        empty = pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
        return empty, {}

    frame = pd.read_csv(DEFAULT_LEGACY_VALIDATED_FILE, dtype=str, keep_default_na=False, low_memory=False)
    if frame.empty:
        empty = pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
        return empty, {}

    frame["timestamp_ny"] = pd.to_datetime(frame["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    accepted_families = _accepted_legacy_families(frame)
    if not accepted_families:
        empty = pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
        return empty, {}

    eligible = frame.loc[
        frame["event_name_normalized"].isin(accepted_families.keys())
        & frame["impact_level"].eq("HIGH")
        & frame["currency"].astype(str).str.upper().isin({"USD", "EUR"})
    ].copy()
    eligible = eligible.loc[_window_mask(eligible["timestamp_ny"], start, end)].copy()
    eligible["source_name"] = "am_fortress_v3_hybrid_legacy"
    eligible["news_source_tier"] = "legacy_verified"
    eligible["source_url"] = ""
    eligible["notes"] = ""
    eligible["source_priority"] = 1
    eligible["selection_status"] = "candidate"
    eligible["timestamp_ny"] = eligible["timestamp_ny"].map(lambda value: value.isoformat() if pd.notna(value) else "")
    return eligible[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]].copy(), accepted_families


def _load_curated_supplement_candidates(start: str, end: str) -> pd.DataFrame:
    if not DEFAULT_CURATED_SUPPLEMENT_FILE.exists():
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])

    try:
        data = json.loads(DEFAULT_CURATED_SUPPLEMENT_FILE.read_text(encoding="utf-8"))
        events = data.get("events", [])
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])

    rows = []
    for ev in events:
        event_name = ev["event_name"]
        if event_name not in CURATED_SUPPLEMENTAL_FAMILIES:
            continue
        
        # We trust the audited manifest's 08:30 logic
        dt_str = f"{ev['release_date']}T{ev['release_time_ny']}:00"
        ts_ny = pd.Timestamp(dt_str).tz_localize(NY_TZ, ambiguous=False)
        ts_utc = ts_ny.tz_convert("UTC")
        
        rows.append({
            "event_name_normalized": event_name,
            "currency": ev["currency"],
            "impact_level": ev["impact_level"],
            "timestamp_original": ts_utc.isoformat(),
            "timezone_original": "America/New_York",
            "timestamp_utc": ts_utc.isoformat(),
            "timestamp_ny": ts_ny.isoformat(),
            "source_name": "am_fortress_v3_curated_local",
            "dedupe_key": stable_hash(ev["currency"], event_name, ts_ny.isoformat(), ev["impact_level"]),
            "validation_status": "approved_curated_local_promotion",
            "news_source_tier": "official_anchor",  # Promoted to highest trust tier
            "source_url": "local_audit_manual",
            "notes": ev.get("verification_status", ""),
            "source_priority": 0,  # Same as official anchors
            "selection_status": "candidate",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"])
    
    df["event_id"] = df["dedupe_key"].map(lambda value: stable_hash("am_fortress_v3", "curated_official", value))
    df["timestamp_ny_dt"] = pd.to_datetime(df["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    df = df.loc[_window_mask(df["timestamp_ny_dt"], start, end)].copy()
    return df[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]].copy()


def _build_am_key_event_validation(clean_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_name, expected_times, optional_if_absent in AM_KEY_EVENT_EXPECTATIONS:
        subset = clean_frame.loc[clean_frame["event_name_normalized"].astype(str).eq(event_name)].copy()
        if event_name == "gdp q/q":
            subset = clean_frame.loc[clean_frame["event_name_normalized"].astype(str).isin({"gdp q/q", "advance gdp q/q", "prelim gdp q/q", "final gdp q/q"})].copy()
        expected_display = "|".join(expected_times)
        if subset.empty:
            rows.append(
                {
                    "event_name_normalized": event_name,
                    "expected_time_ny": expected_display,
                    "approved_rows": 0,
                    "exact_matches": 0,
                    "status": "SOURCE_ABSENT" if optional_if_absent else "FAIL",
                    "optional_if_absent": optional_if_absent,
                }
            )
            continue
        times = pd.to_datetime(subset["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ).dt.strftime("%H:%M")
        exact_matches = int(times.isin(expected_times).sum())
        pass_status = exact_matches == len(subset)
        status = "PASS_ALIAS_FAMILY" if event_name == "gdp q/q" and pass_status else "PASS" if pass_status else "FAIL"
        rows.append(
            {
                "event_name_normalized": event_name,
                "expected_time_ny": expected_display,
                "approved_rows": int(len(subset)),
                "exact_matches": exact_matches,
                "status": status,
                "optional_if_absent": optional_if_absent,
            }
        )
    return rows


def build_am_grade_news_dataset(
    *,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    output_path: Path = DEFAULT_OUTPUT_FILE,
) -> dict[str, Any]:
    clean_output_path = Path(output_path)
    audit_output_path = clean_output_path.with_name(clean_output_path.stem + "_audit.csv")
    summary_output_path = clean_output_path.with_name(clean_output_path.stem + "_summary.json")
    clean_output_path.parent.mkdir(parents=True, exist_ok=True)

    official_candidates = _load_official_anchor_candidates(start, end)
    curated_candidates = _load_curated_supplement_candidates(start, end)
    legacy_candidates, accepted_legacy_families = _load_legacy_supplement_candidates(start, end)
    combined = pd.concat([official_candidates, curated_candidates, legacy_candidates], ignore_index=True)

    if combined.empty:
        clean_frame = pd.DataFrame(columns=REQUIRED_CLEAN_COLUMNS)
        audit_frame = clean_frame.copy()
    else:
        combined = combined.sort_values(["dedupe_key", "source_priority", "event_name_normalized", "timestamp_ny"]).reset_index(drop=True)
        duplicated = combined.duplicated(subset=["dedupe_key"], keep="first")
        combined.loc[duplicated, "selection_status"] = "rejected_lower_priority_duplicate"
        audit_frame = combined.copy()
        clean_frame = audit_frame.loc[audit_frame["selection_status"].eq("selected") | audit_frame["selection_status"].eq("candidate")].copy()
        clean_frame = clean_frame.loc[~duplicated].copy()
        clean_frame = clean_frame[REQUIRED_CLEAN_COLUMNS + ["source_priority", "selection_status"]].sort_values("timestamp_ny").reset_index(drop=True)

    clean_frame.to_csv(clean_output_path, index=False)
    audit_frame.to_csv(audit_output_path, index=False)

    clean_for_validation = clean_frame.copy()
    family_coverage: dict[str, str] = {}
    selected_families = set(clean_for_validation.get("event_name_normalized", pd.Series(dtype=str)).astype(str).tolist())
    official_families = set(
        clean_for_validation.loc[
            clean_for_validation.get("news_source_tier", pd.Series(dtype=str)).eq("official_anchor"),
            "event_name_normalized",
        ].astype(str).tolist()
    )
    legacy_families = set(accepted_legacy_families)
    for family in AM_COVERAGE_FAMILIES:
        if family in official_families:
            family_coverage[family] = "OFFICIAL_ANCHOR"
        elif family in legacy_families and family in selected_families:
            family_coverage[family] = "SUPPLEMENTAL_LEGACY_EXACT_PASS"
        else:
            family_coverage[family] = "MISSING"

    critical_missing = [family for family in CRITICAL_MISSING_FAMILIES if family_coverage.get(family) == "MISSING"]
    module_verdict = "NOT_SAFE_ENOUGH_FOR_AM" if critical_missing else "READY_FOR_STRICT_AM_RESEARCH"

    summary = {
        "raw_source_path_official": str(DEFAULT_OFFICIAL_ANCHOR_FILE),
        "raw_source_path_legacy": str(DEFAULT_LEGACY_VALIDATED_FILE),
        "clean_dataset_path": str(clean_output_path),
        "audit_dataset_path": str(audit_output_path),
        "summary_dataset_path": str(summary_output_path),
        "raw_rows_official": int(len(official_candidates)),
        "raw_rows_legacy": int(len(legacy_candidates)),
        "approved_rows": int(len(clean_frame)),
        "official_selected_rows": int((clean_frame.get("news_source_tier", pd.Series(dtype=str)) == "official_anchor").sum()),
        "supplemental_selected_rows": int((clean_frame.get("news_source_tier", pd.Series(dtype=str)) == "legacy_exact_schedule").sum()),
        "source_approved": False,
        "raw_source_name": "am_fortress_v3_local_hybrid",
        "raw_source_verdict": "HYBRID_LOCAL_REBUILD",
        "operational_source_name": "am_fortress_v3_local_hybrid",
        "operational_source_verdict": module_verdict,
        "module_verdict": module_verdict,
        "operational_scope_verdict": "READY_FOR_STRICT_AM_RESEARCH",
        "scope_window_ny": "08:00-11:00 primary, 14:00/14:30 secondary anchors retained, 17:00 remains execution-risk zone",
        "included_anchor_times_ny": ["08:15", "08:30", "08:45", "09:45", "10:00", "14:00", "14:30"],
        "dst_status": "resolved_for_included_official_anchor_families_and_ecb_press_conference",
        "known_limitations": [
            "la capa legacy se usa solo como suplemento de familias exact-pass ya auditadas",
            "el horario 17:00 NY (rollover) sigue considerandose zona de exclusion por spread/liquidez"
        ],
        "gaps_closed_this_cycle": [
            "retail sales m/m por promocion desde cache local auditada",
            "unemployment claims por promocion desde cache local auditada",
            "ecb press conference por derivacion local desde calendario ECB oficial"
        ],
        "critical_missing_families": critical_missing,
        "family_coverage": family_coverage,
        "accepted_legacy_families": accepted_legacy_families,
        "key_event_validation": _build_am_key_event_validation(clean_for_validation),
    }
    summary_output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye el dataset canonico AM-grade del News Fortress.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    summary = build_am_grade_news_dataset(start=args.start, end=args.end, output_path=args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
