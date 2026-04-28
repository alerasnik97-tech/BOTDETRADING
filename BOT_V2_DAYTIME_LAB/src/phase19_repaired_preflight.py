from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
MANIFEST = LAB / "data" / "certified_data_paths.json"
NEWS_AUDIT = LAB / "outputs" / "m3_bid_ask_certification" / "news_guard" / "news_guard_strict_audit.json"
REQUIRED_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
REQUIRED_END = datetime(2026, 1, 1, tzinfo=timezone.utc)


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
    for key in ["m3_bid", "m3_ask", "m3_spread"]:
        entry = period.get(key)
        if not entry:
            blockers.append(f"{key.upper()}_MISSING")
            continue
        status = _entry_status(entry)
        if status not in {"M3_BID_ASK_CERTIFIED", "M3_CERTIFIED_WITH_WARNINGS"}:
            blockers.append(f"{key.upper()}_NOT_CERTIFIED")
        path = _entry_path(entry)
        if not path or not Path(path).exists():
            blockers.append(f"{key.upper()}_PATH_MISSING")
        source_type = entry.get("source_type") if isinstance(entry, dict) else None
        if source_type != "m1_derived":
            blockers.append(f"{key.upper()}_SOURCE_NOT_M1_DERIVED")
        start = _parse_utc(entry.get("start")) if isinstance(entry, dict) else None
        end = _parse_utc(entry.get("end")) if isinstance(entry, dict) else None
        if start is None or start > REQUIRED_START:
            blockers.append(f"{key.upper()}_COVERAGE_START_INCOMPLETE")
        if end is None or end < REQUIRED_END:
            blockers.append(f"{key.upper()}_COVERAGE_END_INCOMPLETE")
        source = str(entry.get("source", "")) if isinstance(entry, dict) else ""
        if "M5" in source.upper() or "SYNTH" in source.upper():
            blockers.append(f"{key.upper()}_FORBIDDEN_SOURCE")
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
    return {
        "verdict": "PHASE19_REPAIRED_PREFLIGHT_PASSED" if not blockers else "PHASE19_REPAIRED_PREFLIGHT_BLOCKED",
        "blockers": blockers,
        "phase19_not_run": True,
        "mt5_touched": False,
        "real_trading_enabled": False,
    }
