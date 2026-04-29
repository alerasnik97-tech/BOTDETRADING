from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_support import MANIPULANTE, OUT, NY, parse_dt, write_csv, write_json, write_text
from phase37c_mt5_terminal_autodetect import autodetect


PHASE_OUT = OUT.parent / "phase37c_full_auto_ftmo_trial_bootstrap"
LOCAL_CACHE = MANIPULANTE / "09_COMPLIANCE" / "live_news_cache"


def _load(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), "ok"
    except Exception as exc:
        return None, f"malformed:{exc}"


def _events(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    raw = payload.get("events", [])
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        utc_raw = item.get("event_time_utc") or item.get("time_utc")
        event_time_ny = ""
        try:
            event_time_ny = parse_dt(str(utc_raw)).astimezone(NY).isoformat()
        except Exception:
            pass
        rows.append(
            {
                "event_id": item.get("event_id", item.get("id", "")),
                "event_name": item.get("event_name", item.get("name", "")),
                "currency": item.get("currency", ""),
                "impact": item.get("impact", item.get("importance", "")),
                "event_time_utc": utc_raw or "",
                "event_time_ny": event_time_ny,
                "source": payload.get("source_type", payload.get("source", "")),
            }
        )
    return rows


def validate_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"valid": False, "reason": "missing"}
    source = str(payload.get("source_type", payload.get("source", ""))).upper()
    if source not in {"MT5_MQL5_CALENDAR", "MT5_MQL5_ECONOMIC_CALENDAR"}:
        return {"valid": False, "reason": f"bad_source:{source}"}
    generated_raw = payload.get("generated_at_utc")
    if not generated_raw:
        return {"valid": False, "reason": "missing_generated_at"}
    try:
        generated = parse_dt(str(generated_raw))
    except Exception as exc:
        return {"valid": False, "reason": f"timezone_error:{exc}"}
    age = (datetime.now(timezone.utc) - generated).total_seconds() / 60.0
    if age > 60:
        return {"valid": False, "reason": f"cache_stale:{age:.1f}m", "age_minutes": round(age, 3)}
    rows = _events(payload)
    bad_impact = [row for row in rows if str(row["impact"]).upper() not in {"HIGH", "CALENDAR_IMPORTANCE_HIGH", "3"}]
    bad_currency = [row for row in rows if str(row["currency"]).upper() not in {"EUR", "USD"}]
    missing_time = [row for row in rows if not row["event_time_utc"]]
    if bad_impact:
        return {"valid": False, "reason": "unknown_or_non_high_impact"}
    if bad_currency:
        return {"valid": False, "reason": "non_eur_usd_currency"}
    if missing_time:
        return {"valid": False, "reason": "missing_event_time"}
    return {"valid": True, "reason": "OK", "age_minutes": round(age, 3), "event_count": len(rows)}


def validate_cache() -> dict[str, Any]:
    auto = autodetect()
    data_path = Path(str(auto.get("data_path") or ""))
    mt5_files = data_path / "MQL5" / "Files" / "MANIPULANTE"
    source_paths = {
        "today": mt5_files / "ftmo_news_today.json",
        "week": mt5_files / "ftmo_news_week.json",
        "status": mt5_files / "ftmo_news_gate_status.json",
    }
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    date_key = datetime.now(NY).strftime("%Y-%m-%d")
    for key, path in source_paths.items():
        if path.exists():
            dest = LOCAL_CACHE / f"{date_key}_ftmo_news_{'gate_status' if key == 'status' else key}.json"
            shutil.copy2(path, dest)
            copied[key] = str(dest)
    today_payload, today_load = _load(LOCAL_CACHE / f"{date_key}_ftmo_news_today.json")
    week_payload, week_load = _load(LOCAL_CACHE / f"{date_key}_ftmo_news_week.json")
    today_validation = validate_payload(today_payload)
    week_validation = validate_payload(week_payload)
    state = "LIVE_NEWS_CACHE_VALID" if today_validation.get("valid") and week_validation.get("valid") else "NO_TRADE_NEWS_CACHE_MISSING"
    rows_today = _events(today_payload)
    rows_week = _events(week_payload)
    write_csv(PHASE_OUT / "live_news_cache_validation" / "phase37c_news_today.csv", rows_today, ["event_id", "event_name", "currency", "impact", "event_time_utc", "event_time_ny", "source"])
    write_csv(PHASE_OUT / "live_news_cache_validation" / "phase37c_news_week.csv", rows_week, ["event_id", "event_name", "currency", "impact", "event_time_utc", "event_time_ny", "source"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "mt5_files_dir": str(mt5_files),
        "source_paths": {k: str(v) for k, v in source_paths.items()},
        "copied_to_local": copied,
        "today_cache_exists": today_payload is not None,
        "week_cache_exists": week_payload is not None,
        "today_load": today_load,
        "week_load": week_load,
        "today_validation": today_validation,
        "week_validation": week_validation,
        "today_events": len(rows_today),
        "week_events": len(rows_week),
    }


def write_outputs() -> dict[str, Any]:
    status = validate_cache()
    write_json(PHASE_OUT / "live_news_cache_validation" / "phase37c_live_news_cache_validation.json", status)
    write_text(
        PHASE_OUT / "live_news_cache_validation" / "phase37c_live_news_cache_validation.md",
        f"""
# Phase37C Live News Cache Validation

- state: {status['state']}
- MT5 files dir: {status['mt5_files_dir']}
- today cache exists: {status['today_cache_exists']}
- week cache exists: {status['week_cache_exists']}
- today validation: {status['today_validation']}
- week validation: {status['week_validation']}
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
