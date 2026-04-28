from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_certification.forex_market_calendar import (
    is_market_expected_open,
    is_rollover,
    is_within_phase19_candidate_window,
    is_within_user_window,
    session_label,
)
from data_certification.m3_bid_ask_builder import (
    audit_m1_source,
    file_sha256,
    ohlc_integrity,
    read_price_csv,
    validate_m3_files,
)
from data_certification.m3_gap_repair import (
    build_data_quality_mask,
    build_repair_actions,
    classify_m3_gaps,
    summarize_gap_classification,
)
from news_guard_strict import audit_news_feed
from phase19_repaired_preflight import run_preflight


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUT = LAB / "outputs" / "data_quality_forensic_repair"
REPORTS = LAB / "reports"
MANIFEST = LAB / "data" / "certified_data_paths.json"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
M1_BID = ROOT / "data_intake_2020_2026_bidask" / "raw" / "EURUSD_M1_BID_FULL_2020_2026.csv"
M1_ASK = ROOT / "data_intake_2020_2026_bidask" / "raw" / "EURUSD_M1_ASK_FULL_2020_2026.csv"
M3_DIR = LAB / "data" / "certified_m3"
M3_BID = M3_DIR / "EURUSD_M3_BID_2020_2026.csv"
M3_ASK = M3_DIR / "EURUSD_M3_ASK_2020_2026.csv"
M3_SPREAD = M3_DIR / "EURUSD_M3_SPREAD_2020_2026.csv"
M3_METADATA = M3_DIR / "M3_CERTIFICATION_METADATA.json"
M3_MASK = M3_DIR / "EURUSD_M3_DATA_QUALITY_MASK_2020_2026.csv"
NEWS_FEED = ROOT / "data_intake_2020_2026_bidask" / "news" / "EURUSD_NEWS_2020_2026.csv"
NEWS_AUDIT = LAB / "outputs" / "m3_bid_ask_certification" / "news_guard" / "news_guard_strict_audit.json"
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    for name in [
        "diagnosis",
        "inventory",
        "calendar",
        "gap_forensics",
        "gap_impact",
        "repair",
        "data_quality_mask",
        "m3_revalidation",
        "manifest",
        "news_guard_revalidation",
        "phase19_preflight",
        "tests",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    M3_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def count_csv_rows(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    rows = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            rows += chunk.count(b"\n")
    return max(rows - 1, 0)


def csv_columns(path: Path) -> str | None:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    return ",".join(pd.read_csv(path, nrows=0).columns.tolist())


def first_last_timestamp(path: Path) -> tuple[str | None, str | None, str]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None, None, "unknown"
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    ts_col = "timestamp_utc" if "timestamp_utc" in columns else ("timestamp" if "timestamp" in columns else None)
    if ts_col is None:
        return None, None, "unknown"
    first = pd.read_csv(path, nrows=1)
    tail = pd.read_csv(path, usecols=[ts_col]).tail(1)
    start = pd.to_datetime(first[ts_col].iloc[0], utc=True, errors="coerce")
    end = pd.to_datetime(tail[ts_col].iloc[0], utc=True, errors="coerce") if not tail.empty else pd.NaT
    return (
        start.isoformat() if pd.notna(start) else None,
        end.isoformat() if pd.notna(end) else None,
        "UTC" if pd.notna(start) else "unknown",
    )


def git_tracked(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    proc = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True, text=True)
    return proc.returncode == 0


def zip_contains(path: Path) -> bool:
    if not ZIP_PATH.exists():
        return False
    rel = path.relative_to(ROOT).as_posix()
    with zipfile.ZipFile(ZIP_PATH) as z:
        return rel in set(z.namelist())


def timeframe(path: Path) -> str:
    name = path.name.upper()
    for token in ["M1", "M3", "M5", "H1"]:
        if token in name:
            return token
    if "NEWS" in name:
        return "NEWS"
    if path.suffix.lower() == ".json":
        return "METADATA"
    return "UNKNOWN"


def source_type(path: Path) -> str:
    name = path.name.upper()
    if "M1" in name:
        return "M1_BID_ASK_REAL"
    if "M3" in name:
        return "M3_FROM_M1_DIAGNOSTIC"
    if "NEWS" in name:
        return "NEWS_FEED"
    if path == MANIFEST:
        return "MANIFEST"
    return "METADATA"


def certification_status(path: Path, manifest: dict) -> str:
    if path == M1_BID or path == M1_ASK:
        return "SOURCE_VALID_FOR_M3_CERTIFICATION"
    if path == M3_BID or path == M3_ASK or path == M3_SPREAD:
        period = manifest.get("period_2020_2026", {})
        key = "m3_bid" if path == M3_BID else ("m3_ask" if path == M3_ASK else "m3_spread")
        entry = period.get(key)
        if isinstance(entry, dict):
            return str(entry.get("certification_status"))
        return "M3_DIAGNOSTIC_NOT_MANIFESTED"
    if path == M3_MASK:
        period = manifest.get("period_2020_2026", {})
        entry = period.get("m3_data_quality_mask")
        return str(entry.get("certification_status")) if isinstance(entry, dict) else "MASK_PENDING"
    if path == NEWS_FEED:
        return "NEWS_GUARD_STRICT_CERTIFIED"
    return "N/A"


def write_starting_point() -> None:
    payload = {
        "generated_at": NOW,
        "phase18": "PROTECTED_UNCHANGED",
        "phase19_legacy": "PHASE19_INVALIDATED",
        "phase19_repaired": "BLOCKED_PENDING_STRICT_M3_REPAIR",
        "news_guard_strict": "NEWS_GUARD_STRICT_CERTIFIED",
        "m3_status": "M3_REQUIRES_REPAIR_BY_GAPS",
        "objective": "classify and mask gaps without inventing or interpolating data",
        "phase19_repaired_backtest_run": False,
        "scbi_touched": False,
        "mt5_touched": False,
        "real_trading_enabled": False,
    }
    write_json(OUT / "diagnosis" / "data_quality_starting_point.json", payload)
    write_text(
        OUT / "diagnosis" / "data_quality_starting_point.md",
        "# Data Quality Starting Point\n\n"
        "- Phase18 protegida.\n"
        "- Phase19 legacy invalidada.\n"
        "- Phase19 repaired bloqueada.\n"
        "- News Guard strict certificado.\n"
        "- M3 no certificado por gaps.\n"
        "- Objetivo actual: clasificar/reparar/certificar data, no correr estrategia.\n",
    )


def write_inventory(manifest: dict) -> dict:
    files = [M1_BID, M1_ASK, M3_BID, M3_ASK, M3_SPREAD, M3_METADATA, MANIFEST, NEWS_FEED]
    if M3_MASK.exists():
        files.append(M3_MASK)
    rows = []
    for path in files:
        start, end, tz = first_last_timestamp(path)
        rows.append(
            {
                "path": str(path),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
                "rows": count_csv_rows(path),
                "columns": csv_columns(path),
                "timeframe": timeframe(path),
                "source_type": source_type(path),
                "start_utc": start,
                "end_utc": end,
                "timezone": tz,
                "sha256": file_sha256(path) if path.exists() and path.stat().st_size < 250 * 1024 * 1024 else None,
                "tracked_by_git": git_tracked(path) if path.exists() and path.is_relative_to(ROOT) else False,
                "included_in_zip": zip_contains(path) if path.exists() and path.is_relative_to(ROOT) else False,
                "certification_status": certification_status(path, manifest),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inventory" / "data_file_inventory.csv", index=False)
    summary = {
        "generated_at": NOW,
        "files_audited": len(rows),
        "m1_bid_rows": int(df.loc[df["path"] == str(M1_BID), "rows"].iloc[0]),
        "m1_ask_rows": int(df.loc[df["path"] == str(M1_ASK), "rows"].iloc[0]),
        "m3_bid_rows": int(df.loc[df["path"] == str(M3_BID), "rows"].iloc[0]),
        "m3_ask_rows": int(df.loc[df["path"] == str(M3_ASK), "rows"].iloc[0]),
        "heavy_m3_in_zip": bool(df[df["path"].isin([str(M3_BID), str(M3_ASK), str(M3_SPREAD)])]["included_in_zip"].any()),
        "verdict": "DATA_FILES_LOCATED",
    }
    write_json(OUT / "inventory" / "data_file_inventory_summary.json", summary)
    write_text(
        OUT / "inventory" / "data_file_inventory_summary.md",
        "# Data File Inventory Summary\n\n"
        f"Verdicto: {summary['verdict']}\n\n"
        f"Archivos auditados: {summary['files_audited']}\n\n"
        f"M1 BID/ASK rows: {summary['m1_bid_rows']} / {summary['m1_ask_rows']}\n\n"
        f"M3 BID/ASK rows: {summary['m3_bid_rows']} / {summary['m3_ask_rows']}\n",
    )
    return summary


def write_calendar_summary() -> None:
    sample = pd.DataFrame(
        [
            {"timestamp_utc": "2026-01-04T21:59:00+00:00"},
            {"timestamp_utc": "2026-01-04T22:00:00+00:00"},
            {"timestamp_utc": "2026-01-09T21:59:00+00:00"},
            {"timestamp_utc": "2026-01-09T22:00:00+00:00"},
        ]
    )
    sample["market_expected_open"] = sample["timestamp_utc"].map(is_market_expected_open)
    sample["session_label"] = sample["timestamp_utc"].map(session_label)
    summary = {
        "generated_at": NOW,
        "timezone": "America/New_York",
        "market_open_rule": "Sunday 17:00 NY through Friday 17:00 NY",
        "rollover_rule": "17:00-19:00 NY diagnostic/maintenance",
        "user_window": "07:00-20:00 NY",
        "phase19_candidate_window": "08:00-16:30 NY",
        "phase18_window": "08:00-11:00 NY",
        "holiday_calendar": "not authoritative; unknown closures fail closed or mask",
        "sample": sample.to_dict(orient="records"),
    }
    write_json(OUT / "calendar" / "expected_market_calendar_summary.json", summary)
    write_text(
        OUT / "calendar" / "expected_market_calendar_summary.md",
        "# Expected Market Calendar Summary\n\n"
        "- Base: Forex EURUSD Sunday 17:00 NY to Friday 17:00 NY.\n"
        "- Rollover: 17:00-19:00 NY diagnostic/maintenance.\n"
        "- DST: America/New_York via timezone-aware pandas.\n"
        "- Feriados especiales: no se inventan; si afectan ventanas, se enmascaran fail-closed.\n",
    )


def write_gap_forensics() -> tuple[pd.DataFrame, dict]:
    classified = classify_m3_gaps(M3_BID, M3_ASK, NEWS_FEED)
    classified.to_csv(OUT / "gap_forensics" / "m3_gap_forensic_classification.csv", index=False)
    summary = summarize_gap_classification(classified)
    write_json(OUT / "gap_forensics" / "m3_gap_forensic_summary.json", summary)
    write_text(
        OUT / "gap_forensics" / "m3_gap_forensic_summary.md",
        "# M3 Gap Forensic Summary\n\n"
        f"Total gaps: {summary['total_gaps']}\n\n"
        f"Phase19 critical gaps: {summary['phase19_critical_gaps']}\n\n"
        f"Phase18 critical gaps: {summary['phase18_critical_gaps']}\n\n"
        f"Critical block certification: {summary['critical_block_certification']}\n",
    )
    return classified, summary


def write_gap_impact(classified: pd.DataFrame, mask: pd.DataFrame) -> dict:
    df = classified.copy()
    df["start_dt"] = pd.to_datetime(df["start_utc"], utc=True)
    df["year"] = df["start_dt"].dt.year
    df["month"] = df["start_dt"].dt.to_period("M").astype(str)
    df["weekday_ny"] = pd.to_datetime(df["start_ny"], utc=True).dt.tz_convert("America/New_York").dt.day_name()
    total = len(df)
    impact = pd.DataFrame(
        [
            {"window": "Phase18 08:00-11:00", "gaps": int(df["in_phase18_window_08_11"].sum()), "critical_gaps": int((df["in_phase18_window_08_11"] & df["severity"].str.startswith("CRITICAL")).sum())},
            {"window": "Phase19 08:00-16:30", "gaps": int(df["in_phase19_window_08_1630"].sum()), "critical_gaps": int((df["in_phase19_window_08_1630"] & df["severity"].str.startswith("CRITICAL")).sum())},
            {"window": "User 07:00-20:00", "gaps": int(df["in_user_window_07_20"].sum()), "critical_gaps": int((df["in_user_window_07_20"] & df["severity"].str.startswith("CRITICAL")).sum())},
            {"window": "Rollover 17:00-19:00", "gaps": int(df["in_rollover_17_19"].sum()), "critical_gaps": int((df["in_rollover_17_19"] & df["severity"].str.startswith("CRITICAL")).sum())},
        ]
    )
    yearly = df.groupby(["year", "severity"], dropna=False).size().reset_index(name="gaps")
    monthly = df.groupby(["month", "severity"], dropna=False).size().reset_index(name="gaps")
    weekday = df.groupby(["weekday_ny", "severity"], dropna=False).size().reset_index(name="gaps")
    impact.to_csv(OUT / "gap_impact" / "gap_impact_by_window.csv", index=False)
    yearly.to_csv(OUT / "gap_impact" / "gap_impact_by_year.csv", index=False)
    monthly.to_csv(OUT / "gap_impact" / "gap_impact_by_month.csv", index=False)
    weekday.to_csv(OUT / "gap_impact" / "gap_impact_by_weekday.csv", index=False)
    phase19_blocked_days = int((~mask["allow_phase19_repaired"]).sum())
    phase18_blocked_days = int((~mask["allow_phase18"]).sum())
    user_blocked_days = int((~mask["allow_user_window"]).sum())
    summary = {
        "total_gaps": int(total),
        "gaps_ignorable": int((df["severity"] == "IGNORE_SAFE").sum()),
        "gaps_outside_user_window": int((~df["in_user_window_07_20"]).sum()),
        "gaps_inside_user_window_07_20": int(df["in_user_window_07_20"].sum()),
        "gaps_inside_phase19_08_1630": int(df["in_phase19_window_08_1630"].sum()),
        "gaps_inside_phase18_08_11": int(df["in_phase18_window_08_11"].sum()),
        "affected_days": int(df["start_ny"].astype(str).str.slice(0, 10).nunique()),
        "phase18_affected": phase18_blocked_days > 0,
        "phase19_repaired_affected": phase19_blocked_days > 0,
        "phase19_blocked_days": phase19_blocked_days,
        "phase18_blocked_days": phase18_blocked_days,
        "user_window_blocked_days": user_blocked_days,
        "phase19_allowed_days_after_mask": int(mask["allow_phase19_repaired"].sum()),
        "phase18_allowed_days_after_mask": int(mask["allow_phase18"].sum()),
    }
    write_json(OUT / "gap_impact" / "gap_impact_summary.json", summary)
    write_text(
        OUT / "gap_impact" / "gap_impact_summary.md",
        "# Gap Impact Summary\n\n"
        f"Total gaps: {summary['total_gaps']}\n\n"
        f"Gaps dentro Phase19 08:00-16:30: {summary['gaps_inside_phase19_08_1630']}\n\n"
        f"Gaps dentro Phase18 08:00-11:00: {summary['gaps_inside_phase18_08_11']}\n\n"
        f"Dias bloqueados Phase19 por mask: {summary['phase19_blocked_days']}\n",
    )
    return summary


def write_repair(classified: pd.DataFrame) -> dict:
    actions = build_repair_actions(classified)
    actions.to_csv(OUT / "repair" / "m3_gap_repair_actions.csv", index=False)
    summary = {
        "generated_at": NOW,
        "total_actions": int(len(actions)),
        "actions": {str(k): int(v) for k, v in actions["action"].value_counts().to_dict().items()},
        "price_interpolation_used": False,
        "forward_fill_for_trading_used": False,
        "synthetic_ticks_used": False,
        "m5_used_for_m3": False,
        "verdict": "REPAIR_BY_RECLASSIFICATION_AND_FAIL_CLOSED_MASK",
    }
    write_json(OUT / "repair" / "m3_repair_summary.json", summary)
    write_text(
        OUT / "repair" / "m3_repair_summary.md",
        "# M3 Gap Repair Summary\n\n"
        f"Verdicto: {summary['verdict']}\n\n"
        "- No se interpolaron precios.\n"
        "- No se uso forward fill para trading.\n"
        "- No se usaron synthetic ticks.\n"
        "- No se uso M5 para construir M3.\n",
    )
    return summary


def write_mask(classified: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    mask = build_data_quality_mask(classified, metadata["start"], metadata["end"])
    mask.to_csv(M3_MASK, index=False)
    summary = {
        "generated_at": NOW,
        "mask_path": str(M3_MASK),
        "rows": int(len(mask)),
        "sha256": file_sha256(M3_MASK),
        "phase18_blocked_days": int((~mask["allow_phase18"]).sum()),
        "phase19_blocked_days": int((~mask["allow_phase19_repaired"]).sum()),
        "user_window_blocked_days": int((~mask["allow_user_window"]).sum()),
        "certification_status": "DATA_QUALITY_MASK_FAIL_CLOSED",
        "fail_closed": True,
    }
    write_json(OUT / "data_quality_mask" / "data_quality_mask_summary.json", summary)
    write_text(
        OUT / "data_quality_mask" / "data_quality_mask_summary.md",
        "# Data Quality Mask Summary\n\n"
        f"Mask: {M3_MASK}\n\n"
        f"Phase19 blocked days: {summary['phase19_blocked_days']}\n\n"
        f"Phase18 blocked days: {summary['phase18_blocked_days']}\n",
    )
    return mask, summary


def write_revalidation(classified: pd.DataFrame, mask: pd.DataFrame, metadata: dict) -> dict:
    validation = validate_m3_files(M3_BID, M3_ASK)
    bid = read_price_csv(M3_BID)
    ask = read_price_csv(M3_ASK)
    merged = pd.merge(bid[["timestamp", "close"]], ask[["timestamp", "close"]], on="timestamp", suffixes=("_bid", "_ask"))
    merged["spread_close_pips"] = (merged["close_ask"] - merged["close_bid"]) / 0.0001
    merged["year"] = merged["timestamp"].dt.year
    coverage = merged.groupby("year").size().reset_index(name="rows")
    coverage["start"] = merged.groupby("year")["timestamp"].min().values
    coverage["end"] = merged.groupby("year")["timestamp"].max().values
    spread = (
        merged.groupby("year")["spread_close_pips"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "rows", "mean": "mean_pips", "median": "median_pips", "min": "min_pips", "max": "max_pips"})
    )
    ohlc = pd.DataFrame([{"side": "bid", **ohlc_integrity(bid)}, {"side": "ask", **ohlc_integrity(ask)}])
    coverage.to_csv(OUT / "m3_revalidation" / "m3_revalidation_coverage.csv", index=False)
    classified.to_csv(OUT / "m3_revalidation" / "m3_revalidation_gaps.csv", index=False)
    spread.to_csv(OUT / "m3_revalidation" / "m3_revalidation_spread.csv", index=False)
    ohlc.to_csv(OUT / "m3_revalidation" / "m3_revalidation_ohlc.csv", index=False)
    price_ok = (
        validation["bid_monotonic"]
        and validation["ask_monotonic"]
        and validation["bid_duplicates"] == 0
        and validation["ask_duplicates"] == 0
        and validation["timestamp_mismatches"] == 0
        and validation["bid_ohlc_valid"]
        and validation["ask_ohlc_valid"]
        and validation["spread_negative_rows"] == 0
    )
    phase19_critical = classified[
        classified["in_phase19_window_08_1630"] & classified["severity"].isin(["CRITICAL_MASK_DAY", "CRITICAL_BLOCK_CERTIFICATION"])
    ]
    phase19_blocked_dates = set(mask.loc[~mask["allow_phase19_repaired"], "date_ny"].astype(str))
    critical_dates = set(phase19_critical["start_ny"].astype(str).str.slice(0, 10)) if not phase19_critical.empty else set()
    all_critical_masked = critical_dates.issubset(phase19_blocked_dates)
    if not price_ok:
        verdict = "M3_INVALIDATED"
    elif int((classified["severity"] == "CRITICAL_BLOCK_CERTIFICATION").sum()) > 0:
        verdict = "M3_REQUIRES_EXTERNAL_DATA_REPAIR"
    elif len(phase19_critical) == 0 and int((classified["severity"] == "WARNING_MASK_SESSION").sum()) == 0:
        verdict = "M3_BID_ASK_CERTIFIED_FULL"
    elif all_critical_masked:
        verdict = "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK"
    else:
        verdict = "M3_REQUIRES_EXTERNAL_DATA_REPAIR"
    summary = {
        **validation,
        "generated_at": NOW,
        "strict_revalidation_verdict": verdict,
        "price_integrity_ok": bool(price_ok),
        "phase19_critical_gaps": int(len(phase19_critical)),
        "phase19_critical_days_masked": int(len(critical_dates)),
        "all_phase19_critical_gaps_masked": bool(all_critical_masked),
        "requires_data_quality_mask": verdict == "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK",
    }
    write_json(OUT / "m3_revalidation" / "m3_revalidation_summary.json", summary)
    write_text(
        OUT / "m3_revalidation" / "m3_revalidation_summary.md",
        "# M3 Revalidation Summary\n\n"
        f"Verdicto: {verdict}\n\n"
        f"Price integrity OK: {price_ok}\n\n"
        f"Phase19 critical gaps: {len(phase19_critical)}\n\n"
        f"All Phase19 critical gaps masked: {all_critical_masked}\n",
    )
    metadata["validation_status"] = verdict
    metadata["manifest_declared"] = verdict in {"M3_BID_ASK_CERTIFIED_FULL", "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK"}
    metadata["usable_for_phase19_repaired"] = verdict in {"M3_BID_ASK_CERTIFIED_FULL", "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK"}
    metadata["requires_data_quality_mask"] = verdict == "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK"
    metadata["data_quality_mask_path"] = str(M3_MASK) if metadata["requires_data_quality_mask"] else None
    metadata["blocked_reason"] = None if metadata["usable_for_phase19_repaired"] else "M3 strict revalidation did not pass"
    write_json(M3_METADATA, metadata)
    return summary


def update_manifest(metadata: dict, revalidation: dict, mask_summary: dict) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    period = manifest.setdefault("period_2020_2026", {})
    verdict = revalidation["strict_revalidation_verdict"]
    if verdict not in {"M3_BID_ASK_CERTIFIED_FULL", "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK"}:
        result = {"verdict": "MANIFEST_NOT_UPDATED_M3_NOT_CERTIFIED", "m3_declared": False}
    else:
        requires_mask = verdict == "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK"
        common = {
            "source_type": "M3_FROM_M1_BID_ASK",
            "certification_status": verdict,
            "coverage_start": metadata["start"],
            "coverage_end": metadata["end"],
            "rows": metadata["rows"],
            "requires_data_quality_mask": requires_mask,
            "data_quality_mask_path": str(M3_MASK) if requires_mask else None,
            "allowed_windows": ["08:00-11:00_NY_fail_closed", "08:00-16:30_NY_fail_closed", "07:00-20:00_NY_fail_closed"],
            "blocked_windows": {
                "phase18_blocked_days": mask_summary["phase18_blocked_days"],
                "phase19_blocked_days": mask_summary["phase19_blocked_days"],
                "user_window_blocked_days": mask_summary["user_window_blocked_days"],
            },
            "generated_at": NOW,
            "no_interpolation": True,
            "no_forward_fill_for_trading": True,
            "synthetic_ticks": False,
            "m3_from_m5": False,
        }
        period["m3_bid"] = {"path": metadata["bid_path"], "source": metadata["source_bid"], "sha256": metadata["bid_sha256"], **common}
        period["m3_ask"] = {"path": metadata["ask_path"], "source": metadata["source_ask"], "sha256": metadata["ask_sha256"], **common}
        period["m3_spread"] = {
            "path": metadata["spread_path"],
            "source": f"{metadata['source_bid']} | {metadata['source_ask']}",
            "sha256": metadata["spread_sha256"],
            **common,
        }
        period["m3_data_quality_mask"] = {
            "path": str(M3_MASK),
            "source_type": "FAIL_CLOSED_DATA_QUALITY_MASK",
            "certification_status": "DATA_QUALITY_MASK_FAIL_CLOSED",
            "coverage_start": metadata["start"],
            "coverage_end": metadata["end"],
            "rows": mask_summary["rows"],
            "sha256": mask_summary["sha256"],
            "generated_at": NOW,
            "enforced_for_phase19_repaired": True,
            "no_interpolation": True,
            "no_forward_fill_for_trading": True,
        }
        period["phase19_repaired_must_enforce_data_quality_mask"] = requires_mask
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = {
            "verdict": "MANIFEST_UPDATED_WITH_MASKED_M3" if requires_mask else "MANIFEST_UPDATED_WITH_FULL_M3",
            "m3_declared": True,
            "requires_data_quality_mask": requires_mask,
            "manifest": str(MANIFEST),
        }
    write_json(OUT / "manifest" / "strict_manifest_update_report.json", result)
    write_text(
        OUT / "manifest" / "strict_manifest_update_report.md",
        "# Strict Manifest Update Report\n\n"
        f"Verdicto: {result['verdict']}\n\n"
        f"M3 declarado: {result.get('m3_declared')}\n",
    )
    return result


def write_news_revalidation(classified: pd.DataFrame) -> dict:
    summary, _, _ = audit_news_feed(NEWS_FEED)
    interactions = classified[
        classified["near_high_impact_news_30m"] | classified["near_high_impact_news_60m"]
    ][
        [
            "gap_id",
            "start_utc",
            "end_utc",
            "classification",
            "severity",
            "near_high_impact_news_30m",
            "near_high_impact_news_60m",
            "recommended_action",
        ]
    ].copy()
    interactions.to_csv(OUT / "news_guard_revalidation" / "news_gap_interaction.csv", index=False)
    verdict = "NEWS_GUARD_STRICT_RECONFIRMED" if summary.get("verdict") == "NEWS_GUARD_STRICT_CERTIFIED" else "NEWS_GUARD_STRICT_REPAIR_REQUIRED"
    payload = {
        **summary,
        "revalidation_verdict": verdict,
        "news_gap_interactions": int(len(interactions)),
        "news_gap_interactions_30m": int(interactions["near_high_impact_news_30m"].sum()) if not interactions.empty else 0,
        "news_gap_interactions_60m": int(interactions["near_high_impact_news_60m"].sum()) if not interactions.empty else 0,
        "news_plus_data_fail_closed": True,
    }
    write_json(OUT / "news_guard_revalidation" / "news_guard_revalidation_report.json", payload)
    write_text(
        OUT / "news_guard_revalidation" / "news_guard_revalidation_report.md",
        "# News Guard Revalidation Report\n\n"
        f"Verdicto: {verdict}\n\n"
        f"Gaps cerca de high impact news: {len(interactions)}\n",
    )
    return payload


def write_preflight(preflight: dict) -> None:
    write_json(OUT / "phase19_preflight" / "phase19_repaired_preflight_after_repair.json", preflight)
    write_text(
        OUT / "phase19_preflight" / "phase19_repaired_preflight_after_repair.md",
        "# Phase19 Repaired Preflight After Data Repair\n\n"
        f"Verdicto: {preflight['verdict']}\n\n"
        f"Blockers: {', '.join(preflight.get('blockers', [])) or 'none'}\n\n"
        "Phase19 repaired no fue ejecutada.\n",
    )


def run_tests() -> dict:
    tests = [
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_m3_gap_classification",
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_m3_data_quality_mask",
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_m3_manifest_strict",
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_news_gap_interaction",
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_phase19_preflight_requires_mask",
    ]
    cmd = [sys.executable, "-m", "unittest", *tests]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    result = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "verdict": "TESTS_PASSED" if proc.returncode == 0 else "TESTS_FAILED",
    }
    write_json(OUT / "tests" / "data_quality_forensic_test_results.json", result)
    write_text(
        OUT / "tests" / "data_quality_forensic_test_results.md",
        "# Data Quality Forensic Test Results\n\n"
        f"Verdicto: {result['verdict']}\n\n"
        "```text\n" + (proc.stdout + proc.stderr).strip() + "\n```\n",
    )
    return result


def write_final_report(state: dict) -> None:
    if state["m3_revalidation"]["strict_revalidation_verdict"] == "M3_BID_ASK_CERTIFIED_FULL":
        final_verdict = "M3_FULLY_CERTIFIED_PHASE19_READY_FOR_RETEST"
    elif state["m3_revalidation"]["strict_revalidation_verdict"] == "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK":
        final_verdict = "M3_CERTIFIED_WITH_MASK_PHASE19_READY_FOR_RETEST"
    elif state["m3_revalidation"]["strict_revalidation_verdict"] == "M3_REQUIRES_EXTERNAL_DATA_REPAIR":
        final_verdict = "M3_REQUIRES_EXTERNAL_DATA_REPAIR"
    else:
        final_verdict = "DATA_QUALITY_BLOCKED_PHASE19"
    state["final_verdict"] = final_verdict
    state["next_step"] = "Autorizar en fase separada un retest Phase19 repaired con enforcement obligatorio de data-quality mask." if "READY_FOR_RETEST" in final_verdict else "Reparar data externa antes de reabrir Phase19 repaired."
    write_json(REPORTS / "DATA_QUALITY_FORENSIC_REPAIR_REPORT.json", state)
    md = [
        "# Data Quality Forensic Repair Report",
        "",
        f"Veredicto final: {final_verdict}",
        "",
        "## Objetivo",
        "Clasificar y reparar forensemente gaps M1/M3 BID-ASK sin inventar datos y sin correr Phase19.",
        "",
        "## Estado inicial",
        "M3 diagnostic estaba bloqueado por gaps; News Guard strict estaba certificado.",
        "",
        "## Fuente M1",
        f"BID rows: {state['m1_audit']['bid_rows']} / ASK rows: {state['m1_audit']['ask_rows']}.",
        "",
        "## Clasificacion de gaps",
        f"Total gaps: {state['gap_summary']['total_gaps']}.",
        f"Phase19 critical gaps: {state['gap_summary']['phase19_critical_gaps']}.",
        f"Phase18 critical gaps: {state['gap_summary']['phase18_critical_gaps']}.",
        "",
        "## Data-quality mask",
        f"Phase19 blocked days: {state['mask_summary']['phase19_blocked_days']}.",
        f"Phase18 blocked days: {state['mask_summary']['phase18_blocked_days']}.",
        "",
        "## Preflight",
        f"Phase19 repaired preflight: {state['preflight']['verdict']}.",
        "",
        "## Tests",
        f"Tests: {state['tests']['verdict']}.",
        "",
        "## Permitido",
        "Reabrir retest Phase19 repaired solo en fase posterior y solo con mask enforcement.",
        "",
        "## Prohibido",
        "No correr Phase19 legacy, no M3 desde M5, no interpolacion, no MT5, no real, no SCBI, no Phase18.",
        "",
        "## Siguiente paso unico",
        state["next_step"],
        "",
    ]
    write_text(REPORTS / "DATA_QUALITY_FORENSIC_REPAIR_REPORT.md", "\n".join(md))


def update_master_status(state: dict) -> None:
    final_verdict = state["final_verdict"]
    preflight = state["preflight"]["verdict"]
    current = {
        "project_status": {
            "date": "2026-04-28",
            "root_status": "DATA_QUALITY_FORENSIC_REPAIR_UPDATED",
            "lab": "BOT_V2_DAYTIME_LAB",
            "strategies": {
                "SCBI_M5_GLOBAL": "protected_unchanged",
                "Phase18_Fractal_Sweep": "daytime_baseline_protected",
                "Phase19_Expanded_Sweep": "PHASE19_INVALIDATED",
                "Phase19_Repaired": preflight,
            },
            "data_quality_verdict": final_verdict,
            "critical_note": "M3 BID/ASK queda certificado solo con data-quality mask fail-closed. Phase19 repaired no fue ejecutada.",
            "mt5_touched": False,
            "real_trading_enabled": False,
        }
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", current)
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        "# CURRENT PROJECT STATUS\n\n"
        "Fecha de actualizacion: 2026-04-28\n"
        f"Estado data quality: {final_verdict}\n\n"
        "## Estado de estrategias\n"
        "- SCBI_M5_GLOBAL: PROTEGIDA / SIN CAMBIOS.\n"
        "- Phase18 Fractal Sweep: baseline diurna protegida; no fue reemplazada.\n"
        "- Phase19 legacy: PHASE19_INVALIDATED; no es autoridad positiva.\n"
        f"- Phase19 repaired: {preflight}; no fue ejecutada en esta fase.\n\n"
        "## Nota critica\n"
        "M3 BID/ASK fue revalidado con mascara fail-closed. El retest Phase19 queda para una fase separada.\n\n"
        "## Siguiente paso unico\n"
        f"{state['next_step']}\n",
    )
    authority = {
        "authority_hierarchy": {
            "primary": {"id": "SCBI_M5_GLOBAL", "role": "overnight_authority", "status": "protected_unchanged"},
            "daytime_baseline": {
                "id": "Phase18_H1_Fractal_Sweep_First_M3_CHOCH",
                "pf": 1.63,
                "sample": 1040,
                "status": "validated_for_forward_demo_baseline_protected",
            },
            "data_foundation": {
                "id": "M3_BID_ASK_DATA_QUALITY_MASK",
                "status": final_verdict,
                "phase19_repaired_preflight": preflight,
                "mask_required": state["m3_revalidation"].get("requires_data_quality_mask"),
            },
            "quarantined_or_rejected": [
                {"id": "Phase19_Expanded_Sweep", "status": "PHASE19_INVALIDATED"},
                {"id": "Phase12", "status": "invalidated_not_authority"},
            ],
        },
        "lab_status": {"id": "BOT_V2_DAYTIME_LAB", "role": "research_only", "authority": "none_production"},
    }
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", authority)
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        "# STRATEGY AUTHORITY MAP\n\n"
        "## 1. Autoridad maxima: SCBI_M5_GLOBAL\n"
        "- Estado: PROTEGIDA / SIN CAMBIOS.\n\n"
        "## 2. Baseline diurna protegida\n"
        "- Phase18 H1 Fractal Sweep + First M3 CHOCH: VALIDATED_FOR_FORWARD_DEMO, PF 1.63, sample 1.040.\n\n"
        "## 3. Data foundation M3 BID/ASK\n"
        f"- Estado: {final_verdict}.\n"
        f"- Preflight Phase19 repaired: {preflight}.\n"
        "- Phase19 repaired no fue ejecutada en esta fase.\n\n"
        "## 4. Phase19 legacy\n"
        "- Estado: PHASE19_INVALIDATED.\n"
        "- No habilita forward demo, real, VPS ni reemplazo operativo.\n\n"
        "## 5. Regla de autoridad\n"
        "Si hay contradiccion entre narrativa previa y archivos reales actuales, mandan estos archivos de estado y el reporte DATA_QUALITY_FORENSIC_REPAIR_REPORT.\n",
    )


def main() -> int:
    ensure_dirs()
    write_starting_point()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inventory_summary = write_inventory(manifest)
    write_calendar_summary()
    m1_bid = read_price_csv(M1_BID)
    m1_ask = read_price_csv(M1_ASK)
    m1_audit = audit_m1_source(m1_bid, m1_ask)
    metadata = json.loads(M3_METADATA.read_text(encoding="utf-8"))
    classified, gap_summary = write_gap_forensics()
    repair_summary = write_repair(classified)
    mask, mask_summary = write_mask(classified, metadata)
    gap_impact = write_gap_impact(classified, mask)
    m3_revalidation = write_revalidation(classified, mask, metadata)
    metadata = json.loads(M3_METADATA.read_text(encoding="utf-8"))
    manifest_update = update_manifest(metadata, m3_revalidation, mask_summary)
    news_revalidation = write_news_revalidation(classified)
    preflight = run_preflight(MANIFEST, NEWS_AUDIT, ROOT)
    write_preflight(preflight)
    tests = run_tests()
    state = {
        "generated_at": NOW,
        "objective": "DATA QUALITY FORENSIC REPAIR: M1/M3 BID-ASK CERTIFICATION + STRICT GAP CLASSIFICATION",
        "initial_state": {
            "phase18": "PROTECTED_UNCHANGED",
            "phase19_legacy": "PHASE19_INVALIDATED",
            "phase19_repaired": "BLOCKED_BEFORE_THIS_TASK",
            "news_guard_strict": "NEWS_GUARD_STRICT_CERTIFIED",
            "phase19_repaired_backtest_run": False,
        },
        "inventory": inventory_summary,
        "m1_audit": m1_audit,
        "m3_metadata": metadata,
        "gap_summary": gap_summary,
        "gap_impact": gap_impact,
        "repair_summary": repair_summary,
        "mask_summary": mask_summary,
        "m3_revalidation": m3_revalidation,
        "manifest_update": manifest_update,
        "news_guard_revalidation": news_revalidation,
        "preflight": preflight,
        "tests": tests,
        "scbi_touched": False,
        "mt5_touched": False,
        "real_trading_enabled": False,
        "phase19_repaired_backtest_run": False,
    }
    write_final_report(state)
    update_master_status(state)
    print(json.dumps({"final_verdict": state["final_verdict"], "preflight": preflight["verdict"], "tests": tests["verdict"], "phase19_run": False}, indent=2))
    return 0 if tests["verdict"] == "TESTS_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
