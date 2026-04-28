from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
MANIFEST = LAB / "data" / "certified_data_paths.json"
NEWS_AUDIT = LAB / "outputs" / "m3_bid_ask_certification" / "news_guard" / "news_guard_strict_audit.json"
REQUIRED_START = datetime(2020, 1, 1, 22, 0, tzinfo=timezone.utc)
REQUIRED_END = datetime(2026, 1, 1, tzinfo=timezone.utc)
FULL_STATUSES = {"M3_BID_ASK_CERTIFIED", "M3_BID_ASK_CERTIFIED_FULL", "M3_CERTIFIED_WITH_WARNINGS"}
MASKED_STATUSES = {"M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK"}
VALID_STATUSES = FULL_STATUSES | MASKED_STATUSES
VALID_SOURCE_TYPES = {"m1_derived", "M3_FROM_M1_BID_ASK"}


def _entry_path(entry):
    if isinstance(entry, dict):
        return entry.get("path")
    return entry


def _entry_status(entry):
    if isinstance(entry, dict):
        return entry.get("certification_status")
    return None


def _parse_utc(value):
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def run_preflight(
    manifest_path: str | Path = MANIFEST,
    news_audit_path: str | Path = NEWS_AUDIT,
    root_path: str | Path = ROOT,
) -> dict:
    manifest_path = Path(manifest_path)
    blockers = []
    if not manifest_path.exists():
        blockers.append("MANIFEST_MISSING")
        return {"verdict": "PHASE19_REPAIRED_PREFLIGHT_BLOCKED", "blockers": blockers}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    period = manifest.get("period_2020_2026", {})
    mask_required = False
    mask_entry = period.get("m3_data_quality_mask")
    for key in ["m3_bid", "m3_ask", "m3_spread"]:
        entry = period.get(key)
        if not entry:
            blockers.append(f"{key.upper()}_MISSING")
            continue
        status = _entry_status(entry)
        if status not in VALID_STATUSES:
            blockers.append(f"{key.upper()}_NOT_CERTIFIED")
        if status in MASKED_STATUSES or (isinstance(entry, dict) and entry.get("requires_data_quality_mask")):
            mask_required = True
        path = _entry_path(entry)
        if not path or not Path(path).exists():
            blockers.append(f"{key.upper()}_PATH_MISSING")
        source_type = entry.get("source_type") if isinstance(entry, dict) else None
        if source_type not in VALID_SOURCE_TYPES:
            blockers.append(f"{key.upper()}_SOURCE_NOT_M1_DERIVED")
        start = _parse_utc(entry.get("coverage_start") or entry.get("start")) if isinstance(entry, dict) else None
        end = _parse_utc(entry.get("coverage_end") or entry.get("end")) if isinstance(entry, dict) else None
        if start is None or start > REQUIRED_START:
            blockers.append(f"{key.upper()}_COVERAGE_START_INCOMPLETE")
        if end is None or end < REQUIRED_END:
            blockers.append(f"{key.upper()}_COVERAGE_END_INCOMPLETE")
        source = str(entry.get("source", "")) if isinstance(entry, dict) else ""
        if "M5" in source.upper() or "SYNTH" in source.upper():
            blockers.append(f"{key.upper()}_FORBIDDEN_SOURCE")
        if isinstance(entry, dict):
            if entry.get("no_interpolation") is not True:
                blockers.append(f"{key.upper()}_INTERPOLATION_FLAG_MISSING")
            if entry.get("no_forward_fill_for_trading") is not True:
                blockers.append(f"{key.upper()}_FORWARD_FILL_FLAG_MISSING")
            if entry.get("synthetic_ticks") is not False:
                blockers.append(f"{key.upper()}_SYNTHETIC_TICKS_FLAG_INVALID")
    if mask_required:
        if not mask_entry:
            blockers.append("M3_DATA_QUALITY_MASK_MISSING")
        elif not isinstance(mask_entry, dict):
            blockers.append("M3_DATA_QUALITY_MASK_INVALID_ENTRY")
        else:
            mask_path = mask_entry.get("path")
            if not mask_path or not Path(mask_path).exists():
                blockers.append("M3_DATA_QUALITY_MASK_PATH_MISSING")
            if mask_entry.get("certification_status") != "DATA_QUALITY_MASK_FAIL_CLOSED":
                blockers.append("M3_DATA_QUALITY_MASK_NOT_FAIL_CLOSED")
            if mask_entry.get("enforced_for_phase19_repaired") is not True:
                blockers.append("M3_DATA_QUALITY_MASK_NOT_ENFORCED")
    if not Path(news_audit_path).exists():
        blockers.append("NEWS_GUARD_AUDIT_MISSING")
    else:
        news = json.loads(Path(news_audit_path).read_text(encoding="utf-8"))
        if news.get("verdict") not in {"NEWS_GUARD_STRICT_CERTIFIED", "NEWS_GUARD_CERTIFIED_WITH_WARNINGS"}:
            blockers.append("NEWS_GUARD_NOT_CERTIFIED")
    forbidden_names = ["mt5_local_config.json", ".env", "secret", "credentials"]
    root_path = Path(root_path)
    for name in forbidden_names:
        if (root_path / name).exists():
            blockers.append(f"FORBIDDEN_LOCAL_FILE_VISIBLE_{name}")
    if blockers:
        verdict = "PHASE19_REPAIRED_PREFLIGHT_BLOCKED"
    elif mask_required:
        verdict = "PHASE19_REPAIRED_PREFLIGHT_PASSED_MASKED"
    else:
        verdict = "PHASE19_REPAIRED_PREFLIGHT_PASSED_FULL"
    return {
        "verdict": verdict,
        "blockers": blockers,
        "mask_required": mask_required,
        "phase19_not_run": True,
        "mt5_touched": False,
        "real_trading_enabled": False,
    }
