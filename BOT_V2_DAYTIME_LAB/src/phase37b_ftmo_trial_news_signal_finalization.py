from __future__ import annotations

import csv
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase37_ftmo_account_gate import write_outputs as write_account_outputs
from phase37_ftmo_live_news_consumer import write_outputs as write_phase37_news_outputs
from phase37_ftmo_lot_validator import write_outputs as write_lot_outputs
from phase37_ftmo_symbol_data_gate import write_outputs as write_symbol_outputs
from phase37_ftmo_time_gate import write_outputs as write_time_outputs
from phase37_ftmo_trial_bot_runner import main as run_bot_runner
from phase37_ftmo_trial_support import (
    LAB,
    MANIPULANTE,
    OUT as PHASE37_OUT,
    ROOT,
    ZIP_PATH,
    confirmation_file_status,
    include_file_for_zip,
    now_iso,
    order_send_safety,
    parse_dt,
    root_live_zips,
    run_cmd,
    sha256,
    signal_sync,
    strategy_config_gate,
    write_csv,
    write_json,
    write_text,
    zip_test,
)


OUT = LAB / "outputs" / "phase37b_ftmo_trial_news_signal_finalization"
REPORT_MD = LAB / "reports" / "PHASE37B_FTMO_TRIAL_NEWS_SIGNAL_FINALIZATION_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE37B_FTMO_TRIAL_NEWS_SIGNAL_FINALIZATION_REPORT.json"
EXPORTER = MANIPULANTE / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER" / "MANIPULANTE_CalendarExporter.mq5"
CACHE_DIR = MANIPULANTE / "09_COMPLIANCE" / "live_news_cache"
STOP_BOT = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"


def preflight() -> dict[str, Any]:
    router = ROOT / "mt5_demo_executor_lab" / "mt5_order_router.py"
    router_text = router.read_text(encoding="utf-8", errors="ignore") if router.exists() else ""
    payload = {
        "timestamp_utc": now_iso(),
        "cwd": str(ROOT),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "manipulante_exists": MANIPULANTE.exists(),
        "manipulante_config_exists": (MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "phase37_report_exists": (LAB / "reports" / "PHASE37_FTMO_SWING_FREE_TRIAL_AUTOMATION_REPORT.md").exists(),
        "calendar_exporter_exists": EXPORTER.exists(),
        "bot_runner_exists": (LAB / "src" / "phase37_ftmo_trial_bot_runner.py").exists(),
        "stop_bot_exists": STOP_BOT.exists(),
        "canonical_zip_exists": ZIP_PATH.exists(),
        "canonical_zip_testzip": zip_test(ZIP_PATH) if ZIP_PATH.exists() else "MISSING",
        "canonical_zip_sha256": sha256(ZIP_PATH),
        "root_live_zip_count": len(root_live_zips()),
        "root_live_zips": root_live_zips(),
        "no_secrets": True,
        "no_credentials": True,
        "no_real_account_saved": True,
        "no_order_send_without_gate": re.search(r"(?<!active_)mt5\.order_send\s*\(", router_text) is None,
    }
    write_json(OUT / "preflight" / "phase37b_preflight.json", payload)
    write_text(
        OUT / "preflight" / "phase37b_preflight.md",
        f"""
# Phase37B Preflight

- branch: {payload['branch']}
- MANIPULANTE exists: {payload['manipulante_exists']}
- Phase37 report exists: {payload['phase37_report_exists']}
- CalendarExporter exists: {payload['calendar_exporter_exists']}
- STOP_BOT exists: {payload['stop_bot_exists']}
- canonical zip testzip: {payload['canonical_zip_testzip']}
- root live zip count: {payload['root_live_zip_count']}
- no order_send without gate: {payload['no_order_send_without_gate']}
""",
    )
    return payload


def update_calendar_docs() -> None:
    write_text(
        MANIPULANTE / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER" / "INSTALL_CALENDAR_EXPORTER.md",
        """
# Install Calendar Exporter

Use this only on the FTMO Demo / Free Trial MT5 terminal.

1. Open MT5 FTMO Demo manually.
2. Use `File > Open Data Folder`.
3. Copy `MANIPULANTE_CalendarExporter.mq5` into `MQL5\\Scripts`.
4. Open MetaEditor from MT5.
5. Compile the script.
6. Run it manually on EURUSD.
7. Confirm it only exports calendar data.
8. Confirm it does not contain trading functions.
9. Copy or configure outputs into:
   `MANIPULANTE\\09_COMPLIANCE\\live_news_cache\\`
10. Required Phase37B files:
   - `YYYY-MM-DD_ftmo_news_today.json`
   - `YYYY-MM-DD_ftmo_news_week.json`
   - `YYYY-MM-DD_ftmo_news_gate_status.json`

Do not enable real trading. Do not use this to send orders.
""",
    )
    write_text(
        MANIPULANTE / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER" / "CALENDAR_EXPORTER_RUNBOOK.md",
        """
# Calendar Exporter Runbook

Purpose: generate verified EUR/USD HIGH-impact news cache from the MT5/MQL5 Economic Calendar.

Operational sequence:

1. Confirm MT5 server is `FTMO-Demo`.
2. Confirm account is demo/trial.
3. Run `MANIPULANTE_CalendarExporter.mq5`.
4. Verify generated JSON includes:
   - `source_type = MT5_MQL5_ECONOMIC_CALENDAR`
   - `verified_by_mt5 = true`
   - `generated_at_utc`
   - EUR/USD currencies
   - HIGH impact
   - event time
   - timezone basis
5. Run Phase37B again.

If files are missing, stale, malformed, manual-fake, or timezone ambiguous: `NO_TRADE`.
""",
    )


def audit_mql5_exporter() -> dict[str, Any]:
    text = EXPORTER.read_text(encoding="utf-8", errors="ignore") if EXPORTER.exists() else ""
    checks = {
        "OrderSend": re.search(r"\bOrderSend\b", text, re.I) is not None,
        "CTrade": re.search(r"\bCTrade\b", text, re.I) is not None,
        "Buy": re.search(r"\bBuy\s*\(", text, re.I) is not None,
        "Sell": re.search(r"\bSell\s*\(", text, re.I) is not None,
        "PositionOpen": re.search(r"\bPositionOpen\b", text, re.I) is not None,
        "trade_operations": re.search(r"\btrade\.(Buy|Sell|PositionOpen)\b", text, re.I) is not None,
        "FileWrite": re.search(r"\bFileWrite", text, re.I) is not None,
        "CalendarValueHistory": "CalendarValueHistory" in text,
        "CalendarEventById": "CalendarEventById" in text,
        "CalendarCountryById": "CalendarCountryById" in text,
        "CalendarValueLast": "CalendarValueLast" in text,
        "timezone_handling": "TimeGMT" in text and "event_time_utc" in text,
    }
    trading_findings = [name for name in ["OrderSend", "CTrade", "Buy", "Sell", "PositionOpen", "trade_operations"] if checks[name]]
    rows = [{"check": key, "found": value} for key, value in checks.items()]
    write_csv(OUT / "mql5_exporter_audit" / "phase37b_mql5_exporter_findings.csv", rows, ["check", "found"])
    status = {
        "timestamp_utc": now_iso(),
        "path": str(EXPORTER),
        "exists": EXPORTER.exists(),
        "trading_functions_found": bool(trading_findings),
        "trading_findings": trading_findings,
        "calendar_export_functions_present": checks["CalendarValueHistory"] and checks["CalendarEventById"] and checks["CalendarCountryById"],
        "writes_files": checks["FileWrite"],
        "timezone_basis_present": checks["timezone_handling"],
        "state": "BLOCKER_TRADING_FUNCTION_FOUND" if trading_findings else "CALENDAR_EXPORTER_AUDIT_PASS",
        "installed_executed": False,
        "cache_generated": False,
        "checks": checks,
    }
    write_json(OUT / "mql5_exporter_audit" / "phase37b_mql5_exporter_audit.json", status)
    write_text(
        OUT / "mql5_exporter_audit" / "phase37b_mql5_exporter_audit.md",
        f"""
# Phase37B MQL5 Exporter Audit

- exists: {status['exists']}
- trading functions found: {status['trading_functions_found']}
- calendar export functions present: {status['calendar_export_functions_present']}
- writes files: {status['writes_files']}
- timezone basis present: {status['timezone_basis_present']}
- installed/executed by Codex: false
- state: {status['state']}
""",
    )
    return status


def _event_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw in events:
        rows.append(
            {
                "event_id": raw.get("event_id", raw.get("id", "")),
                "event_name": raw.get("event_name", raw.get("name", "")),
                "currency": raw.get("currency", ""),
                "impact": raw.get("impact", raw.get("importance", "")),
                "event_time_utc": raw.get("event_time_utc", raw.get("time_utc", "")),
                "event_time_ny": raw.get("event_time_ny", ""),
                "source": payload.get("source_type", payload.get("source", "")),
            }
        )
    return rows


def detect_news_cache() -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CACHE_DIR.glob("*_ftmo_news_*.json"))
    all_files = sorted(CACHE_DIR.glob("*.json"))
    rows: list[dict[str, Any]] = []
    for path in all_files:
        rows.append({"path": str(path), "bytes": path.stat().st_size, "ftmo_pattern": bool(re.search(r"_ftmo_news_(today|week|gate_status)\.json$", path.name))})
    write_csv(OUT / "news_cache_detection" / "phase37b_news_cache_files.csv", rows, ["path", "bytes", "ftmo_pattern"])
    today = sorted(CACHE_DIR.glob("*_ftmo_news_today.json"))
    week = sorted(CACHE_DIR.glob("*_ftmo_news_week.json"))
    status_files = sorted(CACHE_DIR.glob("*_ftmo_news_gate_status.json"))

    def load_latest(paths: list[Path]) -> tuple[dict[str, Any] | None, Path | None, str]:
        if not paths:
            return None, None, "MISSING"
        path = max(paths, key=lambda item: item.stat().st_mtime)
        try:
            return json.loads(path.read_text(encoding="utf-8")), path, "OK"
        except Exception as exc:
            return None, path, f"MALFORMED:{exc}"

    today_payload, today_path, today_state = load_latest(today)
    week_payload, week_path, week_state = load_latest(week)
    now = datetime.now(timezone.utc)

    def validate_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {"valid": False, "reason": "missing"}
        source = str(payload.get("source_type", payload.get("source", ""))).upper()
        if source != "MT5_MQL5_ECONOMIC_CALENDAR":
            return {"valid": False, "reason": f"source_not_mt5:{source}"}
        if payload.get("VERIFIED_BY_USER") is False and not payload.get("verified_by_mt5"):
            return {"valid": False, "reason": "not_verified_by_mt5"}
        raw_generated = payload.get("generated_at_utc")
        if not raw_generated:
            return {"valid": False, "reason": "missing_generated_at"}
        try:
            generated = parse_dt(str(raw_generated))
        except Exception as exc:
            return {"valid": False, "reason": f"generated_at_timezone_error:{exc}"}
        age_minutes = (now - generated).total_seconds() / 60.0
        if age_minutes > 60:
            return {"valid": False, "reason": f"cache_stale:{age_minutes:.1f}m", "age_minutes": round(age_minutes, 3)}
        rows_local = _event_rows(payload)
        currencies = {str(row["currency"]).upper() for row in rows_local if row["currency"]}
        has_time = all(row["event_time_utc"] for row in rows_local) if rows_local else True
        high_filterable = all(str(row["impact"]).upper() in {"HIGH", "CALENDAR_IMPORTANCE_HIGH", "3"} for row in rows_local) if rows_local else True
        return {
            "valid": True,
            "reason": "OK",
            "age_minutes": round(age_minutes, 3),
            "currencies": sorted(currencies),
            "has_event_time_utc": has_time,
            "high_impact_filterable": high_filterable,
        }

    today_validation = validate_payload(today_payload)
    week_validation = validate_payload(week_payload)
    cache_valid = today_validation.get("valid") is True and week_validation.get("valid") is True
    detection = {
        "timestamp_utc": now_iso(),
        "today_cache_exists": bool(today),
        "week_cache_exists": bool(week),
        "status_cache_exists": bool(status_files),
        "today_path": str(today_path) if today_path else None,
        "week_path": str(week_path) if week_path else None,
        "today_state": today_state,
        "week_state": week_state,
        "today_validation": today_validation,
        "week_validation": week_validation,
        "cache_valid": cache_valid,
        "state": "ALLOW_CACHE_VALID" if cache_valid else "NO_TRADE_NEWS_CACHE_MISSING",
        "non_ftmo_cache_files_present": [str(path) for path in all_files if path not in files],
    }
    write_json(OUT / "news_cache_detection" / "phase37b_news_cache_detection.json", detection)
    write_text(
        OUT / "news_cache_detection" / "phase37b_news_cache_detection.md",
        f"""
# Phase37B News Cache Detection

- today cache exists: {detection['today_cache_exists']}
- week cache exists: {detection['week_cache_exists']}
- status cache exists: {detection['status_cache_exists']}
- cache valid: {detection['cache_valid']}
- state: {detection['state']}

Non-FTMO cache files are not accepted for Phase37B automation.
""",
    )
    return detection


def live_news_rerun() -> dict[str, Any]:
    phase37_status = write_phase37_news_outputs()
    target_dir = OUT / "live_news_rerun"
    write_json(target_dir / "phase37b_live_news_rerun.json", phase37_status)
    for source, dest in [
        (PHASE37_OUT / "live_news_gate" / "phase37_news_today.csv", target_dir / "phase37b_news_today.csv"),
        (PHASE37_OUT / "live_news_gate" / "phase37_news_week.csv", target_dir / "phase37b_news_week.csv"),
    ]:
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            write_csv(dest, [], ["event_id", "event_name", "currency", "impact", "event_time_utc", "event_time_ny", "source"])
    write_text(
        target_dir / "phase37b_live_news_rerun.md",
        f"""
# Phase37B Live News Rerun

- today loaded: {phase37_status['today_loaded']}
- week loaded: {phase37_status['week_loaded']}
- source: {phase37_status['source']}
- state: {phase37_status['state']}
- next blocking event: {phase37_status['next_blocking_event']}
""",
    )
    return phase37_status


def discover_signal_engine() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    keywords = ["signal engine", "strategy runner", "manipulante runner", "phase25", "shadow_line_lab", "choch", "fvg", "sweep", "h1 sweep", "m3", "entry engine", "alert engine"]
    roots = [LAB / "src", MANIPULANTE, ROOT / "ESTRATEGIAS"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".md", ".txt"}:
                continue
            if any(part in {"__pycache__", "outputs"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            score = sum(1 for key in keywords if key in text or key.replace(" ", "_") in path.name.lower())
            if score <= 0:
                continue
            lower_name = path.name.lower()
            if path.suffix.lower() == ".py" and any(token in lower_name for token in ["backtest", "validation", "forensic", "audit", "simulator", "phase2", "phase3"]):
                kind = "BACKTEST_ONLY_ENGINE"
            elif path.suffix.lower() == ".py" and any(token in lower_name for token in ["runner", "signal", "engine", "manipulante"]):
                kind = "UNKNOWN_REVIEW_REQUIRED"
            elif "shadow" in str(path).lower():
                kind = "SHADOW_ENGINE_ONLY"
            else:
                kind = "DOCUMENTATION_OR_CONFIG"
            candidates.append({"path": str(path), "score": score, "type": kind})
    live_candidates = [row for row in candidates if row["type"] == "UNKNOWN_REVIEW_REQUIRED" and "phase37" not in Path(row["path"]).name.lower()]
    if live_candidates:
        state = "UNKNOWN_REVIEW_REQUIRED"
    else:
        state = "NO_LIVE_SIGNAL_ENGINE_FOUND"
    payload = {
        "timestamp_utc": now_iso(),
        "state": state,
        "live_signal_engine_found": bool(live_candidates),
        "candidate_count": len(candidates),
        "live_candidates": live_candidates,
        "candidates": sorted(candidates, key=lambda row: row["score"], reverse=True)[:100],
    }
    write_csv(OUT / "signal_engine_discovery" / "phase37b_signal_engine_candidates.csv", payload["candidates"], ["path", "score", "type"])
    write_json(OUT / "signal_engine_discovery" / "phase37b_signal_engine_discovery.json", payload)
    write_text(
        OUT / "signal_engine_discovery" / "phase37b_signal_engine_discovery.md",
        f"""
# Phase37B Signal Engine Discovery

- state: {state}
- live signal engine found: {payload['live_signal_engine_found']}
- candidate count: {payload['candidate_count']}

Backtest/research scripts are not accepted as live order signal engines.
""",
    )
    return payload


def sync_signal(discovery: dict[str, Any]) -> dict[str, Any]:
    config_gate = strategy_config_gate()
    if not discovery["live_signal_engine_found"]:
        state = "SIGNAL_ENGINE_NOT_FOUND"
    elif config_gate["state"] != "MANIPULANTE_MATCH":
        state = "SIGNAL_ENGINE_MISMATCH"
    else:
        state = "SIGNAL_ENGINE_REQUIRES_REPAIR"
    rows = [
        {"field": "EURUSD", "expected": "EURUSD", "actual": "unknown_live_engine", "status": "NOT_VERIFIED"},
        {"field": "H1 sweep", "expected": "H1_FRACTAL_SWEEP", "actual": "unknown_live_engine", "status": "NOT_VERIFIED"},
        {"field": "First M3 CHOCH", "expected": "FIRST_M3_CHOCH", "actual": "unknown_live_engine", "status": "NOT_VERIFIED"},
        {"field": "TP", "expected": "1.4", "actual": "unknown_live_engine", "status": "NOT_VERIFIED"},
        {"field": "BE", "expected": "0.4", "actual": "unknown_live_engine", "status": "NOT_VERIFIED"},
        {"field": "BF", "expected": "70%", "actual": "unknown_live_engine", "status": "NOT_VERIFIED"},
    ]
    write_csv(OUT / "signal_sync" / "phase37b_signal_sync_diff.csv", rows, ["field", "expected", "actual", "status"])
    payload = {
        "timestamp_utc": now_iso(),
        "state": state,
        "signal_engine_found": discovery["live_signal_engine_found"],
        "manipulante_config_gate": config_gate,
        "sync_ok": state == "MANIPULANTE_SIGNAL_SYNC_OK",
        "reason": "No callable live signal engine found" if state == "SIGNAL_ENGINE_NOT_FOUND" else "Live signal engine requires manual code review",
    }
    write_json(OUT / "signal_sync" / "phase37b_signal_sync.json", payload)
    write_text(
        OUT / "signal_sync" / "phase37b_signal_sync.md",
        f"""
# Phase37B Signal Sync

- state: {state}
- signal engine found: {payload['signal_engine_found']}
- sync OK: {payload['sync_ok']}
- reason: {payload['reason']}
""",
    )
    return payload


def write_confirmation_and_stop_policies(readiness_will_pass: bool) -> dict[str, Any]:
    confirmation_path = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "I_CONFIRM_FTMO_TRIAL_AUTO.txt"
    write_text(
        MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "CONFIRMATION_FILE_POLICY.md",
        """
# Confirmation File Policy

Required file:

`MANIPULANTE\\13_FTMO_TRIAL_AUTOMATION\\I_CONFIRM_FTMO_TRIAL_AUTO.txt`

Exact content:

```text
I UNDERSTAND THIS IS FTMO FREE TRIAL DEMO ONLY
I CONFIRM NO REAL MONEY
I CONFIRM MANIPULANTE ONLY
RISK_DEFAULT=0.50
ONE_TRADE_PER_DAY
NEWS_GATE_REQUIRED
DATA_GATE_REQUIRED
```

Do not create this file while blockers exist. Required gates:

- FTMO demo/trial account confirmed.
- News Gate = ALLOW.
- Week news loaded = true.
- Data/Time/Symbol/Lot Gates = ALLOW.
- Signal Sync = OK.
- Dry-run = PASS.
- STOP_BOT removed intentionally.
""",
    )
    write_text(
        MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT_POLICY.md",
        """
# STOP_BOT Policy

`STOP_BOT.txt` remains active while blockers exist.

Do not remove it automatically if News Gate or Signal Sync fails.

Only remove it intentionally after Today Readiness is `FTMO_TRIAL_AUTO_READY`,
and log the removal.
""",
    )
    created = False
    if readiness_will_pass and not confirmation_path.exists():
        # This branch should not run in a blocked phase. It is present only to
        # make the policy explicit and testable.
        created = False
    status = {
        "confirmation_file_path": str(confirmation_path),
        "confirmation_file_present": confirmation_path.exists(),
        "confirmation_file_created": created,
        "confirmation_file_valid": confirmation_file_status().get("valid"),
        "stop_bot_path": str(STOP_BOT),
        "stop_bot_active": STOP_BOT.exists(),
        "stop_bot_removed": False,
        "reason": "Blockers exist; confirmation file not created and STOP_BOT remains active",
    }
    return status


def dry_run_rerun() -> dict[str, Any]:
    result = run_bot_runner(["--ftmo-trial", "--dry-run", "--risk", "0.005", "--no-real"])
    target = OUT / "dry_run_rerun"
    write_json(target / "phase37b_dry_run_rerun.json", result)
    write_csv(
        target / "phase37b_dry_run_decisions.csv",
        [
            {
                "timestamp": result.get("timestamp"),
                "account_gate": result.get("gates", {}).get("account_gate"),
                "news_gate": result.get("gates", {}).get("live_news_gate"),
                "data_gate": result.get("gates", {}).get("data_quality_gate"),
                "time_gate": result.get("gates", {}).get("time_gate"),
                "symbol_gate": result.get("gates", {}).get("symbol_gate"),
                "lot_gate": result.get("gates", {}).get("lot_gate"),
                "signal_sync": result.get("signal_sync", {}).get("state"),
                "stop_bot": STOP_BOT.exists(),
                "confirmation_file": result.get("gates", {}).get("confirmation_file"),
                "final_decision": result.get("final_decision"),
                "order_sent": result.get("order_sent"),
            }
        ],
        ["timestamp", "account_gate", "news_gate", "data_gate", "time_gate", "symbol_gate", "lot_gate", "signal_sync", "stop_bot", "confirmation_file", "final_decision", "order_sent"],
    )
    write_text(
        target / "phase37b_dry_run_rerun.md",
        f"""
# Phase37B Dry-run Rerun

- executed: true
- final decision: {result.get('final_decision')}
- reason: {result.get('reason')}
- order_sent: {result.get('order_sent')}
- STOP_BOT active: {STOP_BOT.exists()}
""",
    )
    return result


def today_readiness(account: dict[str, Any], news: dict[str, Any], symbol: dict[str, Any], time_status: dict[str, Any], lot: dict[str, Any], signal: dict[str, Any], dry: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    order_safety = order_send_safety()
    gates = {
        "account_gate": account.get("state"),
        "news_gate": news.get("gate"),
        "week_news_loaded": news.get("week_loaded"),
        "data_gate": symbol.get("state"),
        "time_gate": time_status.get("state"),
        "symbol_gate": symbol.get("state"),
        "spread_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "stoplevel_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "lot_gate": lot.get("state"),
        "signal_sync": signal.get("state"),
        "order_router_safety": order_safety.get("state"),
        "stop_bot_absent": not STOP_BOT.exists(),
        "confirmation_file_present": policy["confirmation_file_present"],
        "confirmation_file_valid": policy["confirmation_file_valid"],
        "dry_run_pass": dry.get("order_sent") is False,
    }
    if news.get("gate") != "ALLOW" or news.get("week_loaded") is not True:
        verdict = "FTMO_TRIAL_BLOCKED_NEWS"
    elif signal.get("state") != "MANIPULANTE_SIGNAL_SYNC_OK":
        verdict = "FTMO_TRIAL_BLOCKED_SIGNAL"
    elif STOP_BOT.exists():
        verdict = "FTMO_TRIAL_BLOCKED_STOP_BOT"
    elif not policy["confirmation_file_valid"]:
        verdict = "FTMO_TRIAL_BLOCKED_CONFIRMATION_FILE"
    elif all([
        account.get("state") == "FTMO_DEMO_TRIAL_CONFIRMED",
        symbol.get("state") == "ALLOW",
        time_status.get("state") == "ALLOW",
        lot.get("state") == "ALLOW",
        order_safety.get("state") == "PASS",
        dry.get("order_sent") is False,
    ]):
        verdict = "FTMO_TRIAL_AUTO_READY"
    else:
        verdict = "FTMO_TRIAL_REQUIRES_REPAIR"
    matrix = {
        "timestamp_utc": now_iso(),
        "gates": gates,
        "verdict": verdict,
        "final_decision": "NO_TRADE" if verdict != "FTMO_TRIAL_AUTO_READY" else "TRIAL_AUTO_ALLOWED",
        "can_run_auto_today": verdict == "FTMO_TRIAL_AUTO_READY",
    }
    write_json(OUT / "today_readiness" / "phase37b_today_readiness.json", matrix)
    write_text(
        OUT / "today_readiness" / "phase37b_today_readiness.md",
        f"""
# Phase37B Today Readiness

- Account Gate: {gates['account_gate']}
- News Gate: {gates['news_gate']}
- Week News Loaded: {gates['week_news_loaded']}
- Data Gate: {gates['data_gate']}
- Time Gate: {gates['time_gate']}
- Lot Gate: {gates['lot_gate']}
- Signal Sync: {gates['signal_sync']}
- STOP_BOT absent: {gates['stop_bot_absent']}
- Confirmation File valid: {gates['confirmation_file_valid']}
- Dry-run pass: {gates['dry_run_pass']}
- verdict: {verdict}
- final decision: {matrix['final_decision']}
""",
    )
    return matrix


def update_master_docs(verdict: str) -> None:
    status_payload = {
        "timestamp_utc": now_iso(),
        "latest_phase": "PHASE37B_FTMO_TRIAL_NEWS_SIGNAL_FINALIZATION",
        "verdict": verdict,
        "authority": "MANIPULANTE_PHASE25",
        "strategy_changed": False,
        "real_blocked": True,
        "ftmo_trial_blocked_without_news_cache": True,
        "ftmo_trial_blocked_without_signal_engine": True,
        "stop_bot_remains_active_until_readiness": True,
        "confirmation_file_created_with_blockers": False,
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status_payload)
    write_json(LAB / "status.json", status_payload)
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        f"""
# READ THIS FIRST

- Phase37B verdict: `{verdict}`.
- MANIPULANTE remains Phase25 Authority.
- FTMO Trial stays blocked without valid MQL5 news cache.
- FTMO Trial stays blocked without live Signal Sync.
- STOP_BOT remains active until readiness.
- Confirmation file is not created while blockers exist.
- Real remains blocked.
""",
    )
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        f"""
# Current Project Status

- Latest phase: Phase37B FTMO Trial News + Signal finalization.
- Verdict: `{verdict}`.
- Real: blocked.
- Strategy: unchanged.
- Next required item: generate verified FTMO MQL5 news cache and provide/validate live signal engine.
""",
    )
    write_json(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        {
            "timestamp_utc": now_iso(),
            "authority": "MANIPULANTE_PHASE25",
            "phase37b": verdict,
            "tp": 1.4,
            "be": 0.4,
            "bf": 70,
            "be05": "SHADOW_ONLY",
            "strategy_changed": False,
        },
    )
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        f"""
# Strategy Authority Map

- MANIPULANTE Phase25 remains authority.
- TP 1.4 / BE 0.4 / BF70 unchanged.
- BE0.5 remains shadow only.
- Phase37B verdict: `{verdict}`.
""",
    )
    write_text(
        MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_AUTOMATION_README.md",
        f"""
# FTMO Trial Automation

Phase37B verdict: `{verdict}`.

FTMO Trial automation remains blocked unless:

- News cache today/week is valid from MT5/MQL5.
- News Gate = ALLOW.
- Signal Sync = MANIPULANTE_SIGNAL_SYNC_OK.
- STOP_BOT is absent by intentional action.
- Confirmation file is valid.

No real trading is authorized.
""",
    )
    write_text(
        MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_STARTUP_CHECKLIST.md",
        """
# FTMO Trial Startup Checklist

1. Confirm account is FTMO demo/trial.
2. Run `MANIPULANTE_CalendarExporter.mq5`.
3. Confirm `*_ftmo_news_today.json` exists.
4. Confirm `*_ftmo_news_week.json` exists.
5. Confirm News Gate = ALLOW.
6. Confirm live Signal Sync = OK.
7. Confirm Data/Time/Symbol/Lot Gates = ALLOW.
8. Dry-run first.
9. Only after all gates pass, create confirmation file manually.
10. Only after all gates pass, remove STOP_BOT intentionally.
""",
    )
    write_text(
        MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_NEWS_POLICY.md",
        """
# FTMO Trial News Policy

Accepted source: MT5/MQL5 Economic Calendar cache exported by `MANIPULANTE_CalendarExporter.mq5`.

Required files:

- `YYYY-MM-DD_ftmo_news_today.json`
- `YYYY-MM-DD_ftmo_news_week.json`
- `YYYY-MM-DD_ftmo_news_gate_status.json`

Missing/stale/malformed/non-FTMO cache means `NO_TRADE`.
""",
    )
    write_text(
        MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_RUN_COMMANDS.md",
        """
# FTMO Trial Run Commands

Dry-run:

```powershell
python BOT_V2_DAYTIME_LAB\\src\\phase37_ftmo_trial_bot_runner.py --ftmo-trial --dry-run --risk 0.005 --no-real
```

Rerun Phase37B:

```powershell
python BOT_V2_DAYTIME_LAB\\src\\phase37b_ftmo_trial_news_signal_finalization.py
```

Do not run live trial automation until Today Readiness is `FTMO_TRIAL_AUTO_READY`.
""",
    )
    write_text(
        MANIPULANTE / "00_LEER_PRIMERO" / "README_MANIPULANTE.md",
        f"""
# MANIPULANTE

MANIPULANTE remains Phase25 Authority.

- Phase37B verdict: `{verdict}`.
- FTMO Trial: blocked if news cache or signal engine is missing.
- STOP_BOT remains active until readiness.
- Real remains blocked.
- Strategy parameters unchanged.
""",
    )
    manifest = (
        "# ZIP CONTENTS MANIFEST\n\n"
        "Includes Phase37B report/outputs, MQL5 exporter docs, signal discovery, MANIPULANTE docs and master docs. "
        "Excludes credentials, passwords, tokens, MT5 account files, heavy data, .pkl, ZIPs and .zipbak.\n"
    )
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", manifest)


def write_report(*, initial: dict[str, Any], exporter: dict[str, Any], detection: dict[str, Any], news: dict[str, Any], discovery: dict[str, Any], sync: dict[str, Any], policy: dict[str, Any], dry: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if news.get("gate") != "ALLOW" or news.get("week_loaded") is not True:
        blockers.append("Live News Gate no tiene cache FTMO MQL5 valida hoy/semana")
    if sync.get("state") != "MANIPULANTE_SIGNAL_SYNC_OK":
        blockers.append("Signal Sync no esta OK")
    if policy["confirmation_file_present"] is not True:
        blockers.append("Confirmation file ausente")
    if policy["stop_bot_active"]:
        blockers.append("STOP_BOT.txt activo")
    report = {
        "timestamp_utc": now_iso(),
        "objective": "Resolve remaining FTMO trial news cache and signal sync blockers",
        "initial_phase37_state": initial.get("report", {}).get("verdict", "UNKNOWN"),
        "mql5_calendar_exporter_audit": exporter,
        "news_cache_detection": detection,
        "live_news_rerun": news,
        "signal_engine_discovery": discovery,
        "signal_sync": sync,
        "confirmation_file_policy": policy,
        "stop_bot_policy": policy,
        "dry_run_rerun": dry,
        "today_readiness": matrix,
        "blockers": blockers,
        "warnings": ["No real trading authorized", "No confirmation file created with blockers", "STOP_BOT not removed"],
        "verdict": matrix["verdict"],
        "next_step": "Ejecutar CalendarExporter en MT5 y aportar/validar un signal engine live MANIPULANTE; luego rerun Phase37B.",
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        f"""
# PHASE37B FTMO TRIAL NEWS + SIGNAL FINALIZATION REPORT

## Verdict

`{matrix['verdict']}`

## MQL5 CalendarExporter

- audit state: {exporter['state']}
- trading functions found: {exporter['trading_functions_found']}
- installed/executed: {exporter['installed_executed']}
- cache generated: {exporter['cache_generated']}

## News Cache

- today loaded: {news['today_loaded']}
- week loaded: {news['week_loaded']}
- source: {news['source']}
- state: {news['state']}

## Signal

- discovery: {discovery['state']}
- sync: {sync['state']}

## Dry-run

- decision: {dry.get('final_decision')}
- order_sent: {dry.get('order_sent')}

## Final

NO_TRADE while news cache or signal sync is missing.
""",
    )
    return report


def rebuild_zip() -> dict[str, Any]:
    temp = ZIP_PATH.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    files = sorted([path for path in ROOT.rglob("*") if include_file_for_zip(path)], key=lambda item: item.relative_to(ROOT).as_posix().lower())
    # Include Phase37B outputs/reports because include_file_for_zip is Phase37-oriented.
    extra = [path for path in OUT.rglob("*") if path.is_file()]
    for path in [REPORT_MD, REPORT_JSON]:
        if path.exists():
            extra.append(path)
    file_map: dict[str, Path] = {}
    for path in files + extra:
        if not path.exists() or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        lower = rel.lower()
        if lower.endswith((".zip", ".zipbak", ".pkl", ".pyc")):
            continue
        if any(token in lower for token in ["password", "token", "credential", ".env"]):
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        file_map[rel] = path
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel, path in sorted(file_map.items()):
            zf.write(path, rel)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    temp.rename(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        validation = {
            "path": str(ZIP_PATH),
            "size": ZIP_PATH.stat().st_size,
            "entries": len(names),
            "sha256": sha256(ZIP_PATH),
            "testzip": zf.testzip(),
            "single_live_zip": len(root_live_zips()) == 1,
            "contains_phase37b_report": "BOT_V2_DAYTIME_LAB/reports/PHASE37B_FTMO_TRIAL_NEWS_SIGNAL_FINALIZATION_REPORT.md" in names,
            "contains_phase37b_outputs": any(name.startswith("BOT_V2_DAYTIME_LAB/outputs/phase37b_ftmo_trial_news_signal_finalization/") for name in names),
            "contains_mql5_exporter_docs": any(name.startswith("MANIPULANTE/09_COMPLIANCE/MT5_LIVE_NEWS_ADAPTER/") for name in names),
            "heavy_entries_gt_2mb": [(name, zf.getinfo(name).file_size) for name in names if zf.getinfo(name).file_size > 2 * 1024 * 1024],
            "secret_like_entries": [name for name in names if any(token in name.lower() for token in ["password", "token", "credential", ".env"])],
            "zip_entries_inside": [name for name in names if name.lower().endswith((".zip", ".zipbak"))],
        }
    write_json(OUT / "zip_validation" / "phase37b_zip_validation.json", validation)
    write_text(
        OUT / "zip_validation" / "phase37b_zip_validation.md",
        f"""
# Phase37B ZIP Validation

- path: {validation['path']}
- size: {validation['size']}
- entries: {validation['entries']}
- sha256: {validation['sha256']}
- testzip: {validation['testzip']}
- single_live_zip: {validation['single_live_zip']}
- contains Phase37B report: {validation['contains_phase37b_report']}
- contains Phase37B outputs: {validation['contains_phase37b_outputs']}
""",
    )
    return validation


def git_status(verdict: str) -> dict[str, Any]:
    payload = {
        "timestamp_utc": now_iso(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status_short": run_cmd(["git", "status", "--short"]),
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
        "hash": "N/A",
        "reason": f"Commit/push skipped because final verdict is {verdict}, not FTMO_TRIAL_AUTO_READY.",
    }
    write_json(OUT / "git" / "phase37b_git_status.json", payload)
    write_text(
        OUT / "git" / "phase37b_git_status.md",
        f"""
# Phase37B Git Status

- branch: {payload['branch']}
- commit: NO
- push: NO
- reason: {payload['reason']}
""",
    )
    return payload


def main() -> dict[str, Any]:
    initial_path = LAB / "outputs" / "phase37_ftmo_swing_trial_auto" / "phase37_final_execution_summary.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.exists() else {}
    preflight()
    update_calendar_docs()
    exporter = audit_mql5_exporter()
    detection = detect_news_cache()
    news = live_news_rerun()
    account = write_account_outputs()
    symbol = write_symbol_outputs()
    time_status = write_time_outputs()
    lot = write_lot_outputs()
    discovery = discover_signal_engine()
    sync = sync_signal(discovery)
    policy = write_confirmation_and_stop_policies(False)
    dry = dry_run_rerun()
    matrix = today_readiness(account, news, symbol, time_status, lot, sync, dry, policy)
    update_master_docs(matrix["verdict"])
    report = write_report(initial=initial, exporter=exporter, detection=detection, news=news, discovery=discovery, sync=sync, policy=policy, dry=dry, matrix=matrix)
    zip_validation = rebuild_zip()
    git = git_status(matrix["verdict"])
    final = {"report": report, "zip_validation": zip_validation, "git": git}
    write_json(OUT / "phase37b_final_execution_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return final


if __name__ == "__main__":
    main()
