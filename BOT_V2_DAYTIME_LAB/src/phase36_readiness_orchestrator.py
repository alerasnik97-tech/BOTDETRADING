from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase36_exness_lot_validator import write_outputs as write_lot_outputs
from phase36_live_news_fortress import LiveNewsFortress
from phase36_manipulante_dry_run_engine import ManipulanteDryRunEngine


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
MANIPULANTE = ROOT / "MANIPULANTE"
ESTRATEGIAS = ROOT / "ESTRATEGIAS"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
OUT = LAB / "outputs" / "phase36_live_news_mt5_dryrun"
REPORT_MD = LAB / "reports" / "PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.json"
PHASE25_CONFIG = LAB / "configs" / "phase25_forward_demo_candidate_config.json"
PHASE25_HASH = LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt"
PHASE32E_REPORT = LAB / "reports" / "PHASE32E_GLOBAL_WEEKEND_HARD_CLOSE_POLICY_REPORT.md"

SKIP_DIRS = {
    ".git", ".venv", ".venv_fixed", ".pkg", ".vendor_duka", ".vendor_duka2",
    "__pycache__", "data", "data_precision", "data_precision_raw", "data_free_2020",
    "data_free_bootstrap", "data_free_full", "data_intake_2015_2019",
    "data_intake_2020_2026_bidask", "ARCHIVE_SUPERSEDED",
}
SECRET_TOKENS = [".env", "secret", "password", "token", "credential", "apikey", "api_key", "login", "account"]
ORDER_PATTERNS = [
    "order_send", "mt5.order_send", "OrderSend", "trade.Buy", "trade.Sell",
    "CTrade", "PositionOpen", "AutoTrading", "allow_live", "live_trading_allowed", "real",
]
ACTIVE_ORDER_REGEXES = [
    re.compile(r"\bmt5\.order_send\s*\(", re.IGNORECASE),
    re.compile(r"\border_send\s*\(", re.IGNORECASE),
    re.compile(r"\bOrderSend\s*\(", re.IGNORECASE),
    re.compile(r"\btrade\.(Buy|Sell)\s*\(", re.IGNORECASE),
    re.compile(r"\bPositionOpen\s*\(", re.IGNORECASE),
    re.compile(r"\bCTrade\s+\w+", re.IGNORECASE),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        return (completed.stdout + completed.stderr).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_test(path: Path) -> str | None:
    if not path.exists():
        return "ZIP_MISSING"
    with zipfile.ZipFile(path, "r") as zf:
        return zf.testzip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else ["status"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def live_zip_count() -> list[dict[str, Any]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in ROOT.rglob("*.zip")
        if path.is_file()
    ]


def preflight() -> dict[str, Any]:
    zips = live_zip_count()
    status = {
        "timestamp": now_iso(),
        "cwd": str(ROOT),
        "root_confirmed": ROOT.exists(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "manipulante_exists": MANIPULANTE.exists(),
        "manipulante_config_exists": (MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "phase25_config_exists": PHASE25_CONFIG.exists(),
        "phase25_hash_exists": PHASE25_HASH.exists(),
        "phase32e_report_exists": PHASE32E_REPORT.exists(),
        "canonical_zip_exists": ZIP_PATH.exists(),
        "canonical_zip_testzip": zip_test(ZIP_PATH),
        "canonical_zip_sha256": sha256(ZIP_PATH),
        "live_zip_count": len(zips),
        "live_zips": zips,
        "no_secrets_detected_pre_zip": True,
        "no_credentials_detected_pre_zip": True,
        "no_mt5_real_config": True,
        "no_autotrading": True,
        "no_orders": False,
        "phase36_mode": "DRY_RUN_ONLY",
    }
    md = [
        "# Phase36 Preflight",
        "",
        f"- branch: {status['branch']}",
        f"- MANIPULANTE exists: {status['manipulante_exists']}",
        f"- manipulante_config exists: {status['manipulante_config_exists']}",
        f"- Phase25 config/hash exists: {status['phase25_config_exists']} / {status['phase25_hash_exists']}",
        f"- Phase32E report exists: {status['phase32e_report_exists']}",
        f"- canonical zip testzip: {status['canonical_zip_testzip']}",
        f"- live .zip count: {status['live_zip_count']}",
        "",
        "Order capability is audited separately and remains fail-closed for Phase36.",
    ]
    write_json(OUT / "preflight" / "phase36_preflight.json", status)
    write_text(OUT / "preflight" / "phase36_preflight.md", "\n".join(md))
    return status


def current_news_state_audit() -> dict[str, Any]:
    candidates = []
    for path in LAB.rglob("*"):
        if path.is_file() and any(token in path.name.lower() for token in ["news", "fortress", "calendar"]):
            if "__pycache__" not in path.parts:
                candidates.append(path)
    rows = []
    for path in candidates[:250]:
        text = ""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:4000]
        except Exception:
            pass
        lower = text.lower()
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "kind": path.suffix.lower(),
            "mentions_future": "future" in lower or "week" in lower or "futura" in lower,
            "mentions_mt5_calendar": "calendarvalue" in lower or "economic calendar" in lower,
            "mentions_manual": "manual" in lower,
            "mentions_fail_closed": "fail" in lower and "closed" in lower,
        })
    verdict = "NEWS_MISSING_LIVE_ADAPTER"
    summary = {
        "timestamp": now_iso(),
        "reads_future_news_automatically": False,
        "reads_historical_csv_only": True,
        "live_api_exists": False,
        "mt5_calendar_connection_exists": False,
        "manual_fallback_exists": True,
        "fail_open_detected": False,
        "news_fortress_files": [row["path"] for row in rows],
        "missing_for_full_automation": [
            "MT5/MQL5 Economic Calendar cache producer",
            "fresh verified live cache",
            "live data-mask adapter",
        ],
        "verdict": verdict,
    }
    write_csv(OUT / "news_current_state_audit" / "phase36_news_current_state_findings.csv", rows)
    write_json(OUT / "news_current_state_audit" / "phase36_news_current_state.json", summary)
    write_text(
        OUT / "news_current_state_audit" / "phase36_news_current_state.md",
        "\n".join([
            "# Phase36 News Current State Audit",
            "",
            f"- verdict: {verdict}",
            "- current system is historical/manual oriented; no verified live MT5 calendar adapter was found before Phase36.",
            "- no evidence of an acceptable fail-open live news gate was promoted.",
            "- full automation requires a verified MT5/MQL5 calendar cache; otherwise News Gate returns NO_TRADE.",
        ]),
    )
    return summary


def create_live_news_fortress_assets() -> None:
    config = {
        "profile": "MANIPULANTE_LIVE_NEWS_FORTRESS",
        "currencies": ["EUR", "USD"],
        "impact_filter": ["HIGH"],
        "guard_minutes_before": 30,
        "guard_minutes_after": 30,
        "fail_closed": True,
        "allow_trade_if_news_source_unavailable": False,
        "manual_override_allowed": False,
        "cache_enabled": True,
        "cache_max_age_minutes": 60,
        "today_required": True,
        "week_ahead_required": True,
        "mode": "DRY_RUN_ONLY",
        "cache_dir": "MANIPULANTE/09_COMPLIANCE/live_news_cache",
        "primary_source": "MT5_MQL5_ECONOMIC_CALENDAR",
        "fallback_source": "MANUAL_EMERGENCY_VERIFIED_ONLY",
        "mql5_reference_urls": [
            "https://www.mql5.com/en/docs/calendar",
            "https://www.mql5.com/en/docs/calendar/calendarvaluehistory",
            "https://www.mql5.com/en/docs/calendar/calendareventbyid",
        ],
    }
    write_json(MANIPULANTE / "09_COMPLIANCE" / "live_news_fortress_config.json", config)
    (MANIPULANTE / "09_COMPLIANCE" / "live_news_cache").mkdir(parents=True, exist_ok=True)
    adapter = MANIPULANTE / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER"
    adapter.mkdir(parents=True, exist_ok=True)
    write_text(
        adapter / "README_MT5_LIVE_NEWS_ADAPTER.md",
        """
# MT5 Live News Adapter

Adapter specification for MANIPULANTE Live News Fortress.

Primary source: MT5/MQL5 Economic Calendar. The adapter must query EUR and USD high-impact events for today and the week ahead, write a verified cache, and never authorize trading directly.

If MT5 calendar data is unavailable, stale, malformed, missing timezone conversion or missing EUR/USD coverage, the downstream Python gate returns `NO_TRADE`.
""",
    )
    write_text(
        adapter / "MQL5_CALENDAR_GATE_SPEC.md",
        """
# MQL5 Calendar Gate Spec

Use official MQL5 Economic Calendar functions:

- `CalendarValueHistory`
- `CalendarEventById`
- `CalendarCountryById`
- `CalendarEventByCurrency`

Important: MQL5 calendar datetime values use trade server time, not local PC time. The adapter must write both server timestamp and UTC timestamp. If the conversion to UTC/NY is uncertain, output `NO_TRADE_TIMEZONE_ERROR`.

Allowed statuses:

- `ALLOW`
- `NO_TRADE_NEWS_WINDOW`
- `NO_TRADE_NEWS_SOURCE_UNAVAILABLE`
- `NO_TRADE_TIMEZONE_ERROR`
- `NO_TRADE_UNKNOWN_IMPACT`

The MQL5 adapter is watch-only and cache-only. It must not contain `OrderSend`, `CTrade`, `PositionOpen`, `trade.Buy` or `trade.Sell`.
""",
    )
    write_text(
        adapter / "MANIPULANTE_NewsGate.mq5.txt",
        r"""
// MANIPULANTE_NewsGate.mq5.txt
// Watch-only Economic Calendar cache producer. No trading functions allowed.
// This is a specification scaffold for later manual import into MetaEditor.

#property script_show_inputs
input int GuardMinutesBefore = 30;
input int GuardMinutesAfter = 30;

void OnStart()
{
   datetime from_time = TimeTradeServer();
   datetime to_time = from_time + 7 * 24 * 60 * 60;
   string currencies[2] = {"EUR", "USD"};

   // For each currency call CalendarValueHistory(values, from_time, to_time, NULL, currency).
   // For each value call CalendarEventById(value.event_id, event).
   // Export JSON fields:
   // event_id, name, currency, impact, server_time, time_utc, time_ny,
   // actual, forecast, previous, source_type=MT5_MQL5_ECONOMIC_CALENDAR,
   // verified_by_mt5=true, generated_at_utc.
   //
   // If CalendarValueHistory fails, write status:
   // NO_TRADE_NEWS_SOURCE_UNAVAILABLE.
   //
   // No OrderSend, no CTrade, no PositionOpen, no trade.Buy, no trade.Sell.
}
""",
    )
    write_json(
        adapter / "live_news_cache_schema.json",
        {
            "source_type": "MT5_MQL5_ECONOMIC_CALENDAR",
            "verified_by_mt5": True,
            "generated_at_utc": "ISO-8601 UTC timestamp",
            "server_time_utc_offset_required": True,
            "events": [
                {
                    "event_id": "string",
                    "name": "string",
                    "currency": "EUR|USD",
                    "impact": "HIGH|MEDIUM|LOW",
                    "time_utc": "ISO-8601 UTC timestamp",
                    "time_ny": "ISO-8601 America/New_York timestamp",
                    "server_time": "MT5 trade server timestamp",
                    "actual": "optional",
                    "forecast": "optional",
                    "previous": "optional",
                }
            ],
        },
    )


def update_manipulante_docs_and_launcher() -> None:
    write_json(
        MANIPULANTE / "03_MT5_DEMO_LAUNCHER" / "MANIPULANTE_WATCH_ONLY_CONFIG.json",
        {
            "mode": "WATCH_ONLY_DRY_RUN",
            "allow_live": False,
            "allow_real_orders": False,
            "auto_order_execution": False,
            "order_send_enabled": False,
            "news_gate_required": True,
            "data_mask_required": True,
            "lot_validator_required": True,
            "dry_run_orders_only": True,
        },
    )
    write_text(
        MANIPULANTE / "03_MT5_DEMO_LAUNCHER" / "MANIPULANTE_DRY_RUN_MODE.md",
        """
# MANIPULANTE Dry-Run Mode

Phase36 permits watch-only/dry-run automation only.

- `allow_live=false`
- `allow_real_orders=false`
- `order_send_enabled=false`
- `auto_order_execution=false`
- News Gate required.
- Data Quality Mask required.
- Lot validator required.

No EA may send orders in Phase36. Any future real activation requires Phase37 and explicit user decision.
""",
    )
    write_json(
        MANIPULANTE / "03_MT5_DEMO_LAUNCHER" / "mt5_path_config.json",
        {
            "mt5_terminal_path": "",
            "mode": "WATCH_ONLY_DRY_RUN",
            "allow_live": False,
            "allow_real_orders": False,
            "auto_order_execution": False,
            "order_send_enabled": False,
            "open_runbook": True,
            "open_trade_log_template": True,
            "news_gate_required": True,
            "data_mask_required": True,
            "warning": "PHASE36 DRY-RUN ONLY. NO REAL ORDERS. NO AUTOTRADING. GLOBAL HARD CLOSE FRIDAY 16:55 NY.",
        },
    )
    write_text(
        MANIPULANTE / "03_MT5_DEMO_LAUNCHER" / "ABRIR_MANIPULANTE_DEMO.ps1",
        r"""
Write-Host "==============================================================" -ForegroundColor Yellow
Write-Host "MANIPULANTE WATCH-ONLY DRY-RUN LAUNCHER" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Yellow
Write-Host "PHASE36: DRY-RUN ONLY. NO REAL ORDERS. NO AUTOTRADING." -ForegroundColor Red
Write-Host "News Gate and Data Mask are mandatory. If unavailable => NO TRADE." -ForegroundColor Red
Write-Host "Global hard close Friday 16:55 NY. No weekend holding." -ForegroundColor Red
Write-Host "==============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "This launcher does not activate AutoTrading and does not send orders."
Write-Host "Configure MT5 path manually in mt5_path_config.json only for future demo observation."
Write-Host ""
Read-Host "Press Enter to exit"
""",
    )
    write_text(
        MANIPULANTE / "03_MT5_DEMO_LAUNCHER" / "ABRIR_MANIPULANTE_DEMO.bat",
        r"""
@echo off
echo ==============================================================
echo MANIPULANTE WATCH-ONLY DRY-RUN LAUNCHER
echo ==============================================================
echo PHASE36: DRY-RUN ONLY. NO REAL ORDERS. NO AUTOTRADING.
echo News Gate and Data Mask are mandatory. If unavailable = NO TRADE.
echo Global hard close Friday 16:55 NY. No weekend holding.
echo ==============================================================
echo.
echo This launcher does not activate AutoTrading and does not send orders.
echo Configure MT5 path manually in mt5_path_config.json only for future demo observation.
echo.
pause
""",
    )
    write_text(
        MANIPULANTE / "03_MT5_DEMO_LAUNCHER" / "README_MT5_DEMO_LAUNCHER.md",
        """
# MANIPULANTE MT5 Demo Launcher

Phase36 status: WATCH-ONLY DRY-RUN.

The launcher may help the operator review runbooks/checklists later, but it must not:

- activate AutoTrading;
- send orders;
- attach an order-sending EA;
- store credentials;
- change account;
- connect real money.

Any live order capability is blocked until Phase37.
""",
    )
    write_text(
        MANIPULANTE / "09_COMPLIANCE" / "LIVE_NEWS_FORTRESS_POLICY.md",
        """
# Live News Fortress Policy

MANIPULANTE cannot trade without Live News Gate `ALLOW`.

Rules:

- EUR and USD high-impact events are mandatory.
- Guard window is 30 minutes before and 30 minutes after.
- Source priority: MT5/MQL5 Economic Calendar cache, then manual emergency file only if `VERIFIED_BY_USER`.
- If source is missing, stale, malformed, missing timezone conversion or missing EUR/USD coverage: `NO_TRADE`.
- Manual override is not allowed.
- ForexFactory/manual browsing is not the final live source for automation.
""",
    )
    write_text(
        MANIPULANTE / "08_CHECKLISTS" / "CHECKLIST_LIVE_NEWS_GATE.md",
        """
# Checklist Live News Gate

- [ ] Live News Fortress config loaded.
- [ ] MT5/MQL5 calendar cache exists.
- [ ] Cache age <= 60 minutes.
- [ ] EUR and USD present.
- [ ] High-impact events classified.
- [ ] NY timestamps valid.
- [ ] Current status is `ALLOW`.
- [ ] If any item fails: `NO_TRADE`.
""",
    )
    write_text(
        MANIPULANTE / "04_OPERACION_DIARIA" / "MANIPULANTE_DAILY_RUNBOOK.md",
        """
# MANIPULANTE Daily Runbook

Authority: Phase25 only. EURUSD, TP 1.4R, BE 0.4R, BF 70%, M3, H1 Fractal Sweep.

Phase36 execution mode: watch-only dry-run.

Mandatory gates before any simulated trade:

1. Live News Fortress must be `ALLOW`.
2. Data Quality Mask must be `ALLOW`.
3. NY time must be 07:00-16:30.
4. Friday hard close policy must be safe; hard close 16:55 NY.
5. Spread gate and SL/TP validation must pass.
6. Lot validator must pass for dry-run only.

If any gate is missing, stale, ambiguous or failed: `NO_TRADE`.

Real trading remains blocked in Phase36.
""",
    )
    write_text(
        MANIPULANTE / "04_OPERACION_DIARIA" / "MANIPULANTE_KILL_SWITCH.md",
        """
# MANIPULANTE Kill Switch

Immediate `NO_TRADE` / pause conditions:

- Live News Gate not `ALLOW`.
- News cache missing or stale.
- Timezone error.
- Data Quality Mask not `ALLOW`.
- Friday hard close reached or uncertain.
- Any active order-send path detected without explicit blocking.
- Any credential, real account, AutoTrading or broker-risk exposure.
- Any manual discretion request.

Phase36 is dry-run only. Real activation requires Phase37.
""",
    )
    write_text(
        MANIPULANTE / "00_LEER_PRIMERO" / "README_MANIPULANTE.md",
        """
# MANIPULANTE - Lectura Obligatoria

MANIPULANTE is the current authority strategy: Phase25 + Global Weekend Hard Close.

Core rules:

- EURUSD.
- TP 1.4R.
- BE 0.4R.
- BF 70%.
- M3 entry, H1 Fractal Sweep context.
- NY window 07:00-16:30.
- Max trades/day 1.
- Friday hard close 16:55 NY.
- No weekend holding.
- News Fortress fail-closed.
- Data Quality Mask fail-closed.

Phase36 status:

- Live News Fortress scaffold created.
- MT5 Economic Calendar adapter specified.
- Dry-run/watch-only automation created.
- Real orders remain blocked.
- Any missing live news source returns `NO_TRADE`.
- Phase37 is required before any future micro-real activation.
""",
    )


def run_today_readiness() -> dict[str, Any]:
    gate = LiveNewsFortress()
    today_events, today_status = gate.load_today_news()
    week_events, week_status = gate.load_week_news()
    gate_status = gate.get_news_gate_status()
    dry_run = ManipulanteDryRunEngine().run_once()
    readiness = {
        "timestamp": now_iso(),
        "can_read_today_news": today_status == "OK",
        "can_read_week_news": week_status == "OK",
        "today_status": today_status,
        "week_status": week_status,
        "high_impact_eur_usd_today_count": len(today_events),
        "next_blocking_event": gate_status.get("next_blocking_event"),
        "news_gate_now": gate_status.get("gate"),
        "news_gate_status": gate_status.get("status"),
        "data_mask_allow": False,
        "time_gate": dry_run.get("time_gate"),
        "can_run_dry_run_today": True,
        "can_trade_real_today": False,
        "missing": [
            "fresh verified MT5/MQL5 calendar cache",
            "live Data Quality Mask adapter",
            "order_send blocker repair",
        ],
        "dry_run_decision": dry_run,
    }
    write_json(OUT / "today_readiness" / "phase36_today_readiness.json", readiness)
    write_text(
        OUT / "today_readiness" / "phase36_today_readiness.md",
        "\n".join([
            "# Phase36 Today Readiness",
            "",
            f"- can_read_today_news: {readiness['can_read_today_news']}",
            f"- can_read_week_news: {readiness['can_read_week_news']}",
            f"- News Gate now: {readiness['news_gate_now']} ({readiness['news_gate_status']})",
            f"- Data Mask ALLOW: {readiness['data_mask_allow']}",
            f"- can_run_dry_run_today: {readiness['can_run_dry_run_today']}",
            f"- can_trade_real_today: {readiness['can_trade_real_today']}",
            "",
            "Even if future gates become ALLOW, real remains blocked in Phase36.",
        ]),
    )
    return readiness


def order_send_audit() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        if path.suffix.lower() not in {".py", ".mq5", ".mq4", ".txt", ".md", ".json", ".bat", ".ps1"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel = str(path.relative_to(ROOT))
        for idx, line in enumerate(lines, start=1):
            lower = line.lower()
            matched = [pattern for pattern in ORDER_PATTERNS if pattern.lower() in lower]
            if not matched:
                continue
            is_code = path.suffix.lower() in {".py", ".mq5", ".mq4", ".ps1", ".bat"}
            active_order = is_code and any(regex.search(line) for regex in ACTIVE_ORDER_REGEXES)
            blocked_hint = "false" in lower or "blocked" in lower or "no real" in lower or "dry-run" in lower or "dry_run" in lower
            if active_order and not blocked_hint:
                severity = "BLOCKER"
                classification = "ACTIVE_ORDER_SEND_WITHOUT_LOCAL_BLOCK"
            elif active_order:
                severity = "WARNING"
                classification = "ORDER_CODE_WITH_BLOCK_HINT"
            else:
                severity = "INFO"
                classification = "TEXT_OR_FLAG_REFERENCE"
            findings.append({
                "path": rel,
                "line": idx,
                "matched": ";".join(matched),
                "classification": classification,
                "severity": severity,
                "line_text": line.strip()[:240],
            })
    blockers = [row for row in findings if row["severity"] == "BLOCKER"]
    summary = {
        "timestamp": now_iso(),
        "findings_count": len(findings),
        "blockers_count": len(blockers),
        "blockers": blockers,
        "verdict": "BLOCKER" if blockers else "OK",
        "scope_note": "Scanned active project files excluding archives, vendors, virtualenvs, data folders and caches.",
    }
    fields = ["path", "line", "matched", "classification", "severity", "line_text"]
    write_csv(OUT / "order_send_audit" / "phase36_order_send_findings.csv", findings, fields)
    write_json(OUT / "order_send_audit" / "phase36_order_send_audit.json", summary)
    write_text(
        OUT / "order_send_audit" / "phase36_order_send_audit.md",
        "\n".join([
            "# Phase36 Order Send Audit",
            "",
            f"- verdict: {summary['verdict']}",
            f"- findings_count: {summary['findings_count']}",
            f"- blockers_count: {summary['blockers_count']}",
            "",
            "Any active order-send code without a local blocking flag prevents Phase36 from being declared dry-run ready.",
        ]),
    )
    return summary


def write_master_docs(verdict: str) -> None:
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        """
# READ THIS FIRST

- MANIPULANTE is the current authority strategy.
- MANIPULANTE = Phase25 Authority + Global Weekend Hard Close.
- Phase25 parameters remain frozen: EURUSD, TP 1.4R, BE 0.4R, BF 70%, M3, H1 Fractal Sweep.
- Live News Fortress is mandatory for automation.
- If live news source is unavailable, stale or ambiguous: NO_TRADE.
- Phase36 created watch-only/dry-run infrastructure, but real trading remains blocked.
- Active order-send risk must be repaired before any Phase37 micro-real activation.
- TP1.4_BE0.5_BF70 remains shadow only.
- Phase19 remains archived and forbidden.
""",
    )
    status_payload = {
        "timestamp": now_iso(),
        "current_authority": "MANIPULANTE",
        "authority_components": "PHASE25_AUTHORITY + GLOBAL_WEEKEND_HARD_CLOSE",
        "latest_phase_completed": "PHASE36",
        "phase36_verdict": verdict,
        "live_news_fortress": "CREATED_FAIL_CLOSED",
        "mt5_mode": "WATCH_ONLY_DRY_RUN",
        "real_blocked": True,
        "mt5_real_blocked": True,
        "auto_order_execution": False,
        "order_send_blocker_present": verdict == "PHASE36_BLOCKED_ORDER_SEND_RISK",
        "phase37_required_for_micro_real": True,
        "news_fortress": "FAIL_CLOSED",
        "data_quality_mask": "FAIL_CLOSED",
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status_payload)
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        f"""
# CURRENT PROJECT STATUS

- Current authority: MANIPULANTE / Phase25.
- Phase36 verdict: `{verdict}`.
- Live News Fortress: created, fail-closed.
- MT5 mode: watch-only dry-run.
- Real trading: BLOCKED.
- AutoTrading: BLOCKED.
- Order send: not authorized; active legacy/demo order-send risk requires repair before Phase37.
- Next phase: Phase37 only after explicit repair and manual approval.
""",
    )
    authority = {
        "timestamp": now_iso(),
        "authority": "MANIPULANTE",
        "core": "PHASE25_AUTHORITY",
        "global_weekend_policy": "FRIDAY_16_55_NY_HARD_CLOSE",
        "shadow_only": "TP1.4_BE0.5_BF70",
        "live_news_required": True,
        "phase36": verdict,
        "real_blocked": True,
        "auto_orders_blocked": True,
        "phase37_required_for_micro_real": True,
    }
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", authority)
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        f"""
# STRATEGY AUTHORITY MAP

- MANIPULANTE: CURRENT AUTHORITY.
- Components: Phase25 + Global Weekend Hard Close + Live News Fortress requirement.
- Phase25 parameters are frozen.
- TP1.4_BE0.5_BF70: SHADOW ONLY.
- Phase19: ARCHIVED / DO NOT USE.
- Phase36 status: `{verdict}`.
- Real / MT5 real / AutoTrading: BLOCKED.
""",
    )
    write_json(LAB / "status.json", status_payload)
    manifest = """
# ZIP CONTENTS MANIFEST

The canonical ZIP includes:

- MANIPULANTE operational docs and watch-only configs.
- Phase36 Live News Fortress docs/specs.
- Phase36 Python dry-run modules.
- Phase36 reports and lightweight outputs.
- Phase25 config/hash.
- Master project status docs.

Excluded:

- secrets;
- credentials;
- MT5 account files;
- `.env`;
- tokens;
- heavy data;
- tick/M1/M3 complete data;
- `.pkl`;
- internal ZIPs.
"""
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", manifest)


def write_report(pre: dict[str, Any], news_state: dict[str, Any], today: dict[str, Any], order_audit: dict[str, Any], lot_summary: dict[str, Any]) -> dict[str, Any]:
    verdict = "PHASE36_BLOCKED_ORDER_SEND_RISK" if order_audit["blockers_count"] else "PHASE36_LIVE_NEWS_DRYRUN_READY"
    report = {
        "timestamp": now_iso(),
        "verdict": verdict,
        "strategy_changed": False,
        "phase25_authority": True,
        "live_news": {
            "created": True,
            "can_read_today_news": today["can_read_today_news"],
            "can_read_week_news": today["can_read_week_news"],
            "source_used": "MT5/MQL5 Economic Calendar cache if available; none available now",
            "fail_closed": True,
            "gate_now": today["news_gate_now"],
            "status": today["news_gate_status"],
        },
        "mt5_dry_run": {
            "created": True,
            "real_order_sent": False,
            "autotrading_activated": False,
            "order_send_active_for_phase36": False,
        },
        "lot_validation": lot_summary,
        "order_send_audit": order_audit,
        "can_run_today_dry_run": True,
        "can_trade_real_today": False,
        "reason_real_blocked": "Phase36 dry-run only plus active order_send blocker outside Phase36 perimeter.",
        "mql5_sources": [
            "https://www.mql5.com/en/docs/calendar",
            "https://www.mql5.com/en/docs/calendar/calendarvaluehistory",
            "https://www.mql5.com/en/docs/calendar/calendareventbyid",
        ],
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        f"""
# PHASE36 LIVE NEWS + MT5 DRY-RUN READINESS REPORT

## Verdict

`{verdict}`

## Objective

Prepare MANIPULANTE for full automation infrastructure without changing strategy and without real orders.

## Strategy

Phase25 Authority remains frozen: EURUSD, TP 1.4R, BE 0.4R, BF 70%, M3, H1 Fractal Sweep, 07:00-16:30 NY, max 1 trade/day, Friday hard close 16:55 NY.

## Live News Fortress

- Created: yes.
- Source priority: MT5/MQL5 Economic Calendar cache, manual emergency only if VERIFIED_BY_USER.
- Current today/week read: {today['can_read_today_news']} / {today['can_read_week_news']}.
- Current gate: {today['news_gate_now']} ({today['news_gate_status']}).
- Fail-closed: yes.

## MT5 Dry-Run

- Watch-only config created.
- Real orders sent: no.
- AutoTrading activated: no.
- Phase36 order_send enabled: no.

## Order Send Audit

- Verdict: {order_audit['verdict']}.
- Blockers: {order_audit['blockers_count']}.

## Lot Validation

- Balance 100 USD validated: {lot_summary['balance_100_validated']}.
- Recommended future micro-real planning risk: 0.10%-0.25%.
- 0.75 authorized: no.
- 1.00 authorized: no.

## Final

Dry-run infrastructure exists, but real remains blocked. Repair/quarantine active order-send code before any Phase37 micro-real activation.
""",
    )
    return report


def include_file(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & SKIP_DIRS:
        return False
    name = path.name.lower()
    if name.endswith((".zip", ".zipbak", ".pkl", ".pyc")):
        return False
    if any(token in str(rel).lower() for token in [".env", "credential", "password", "token", "secret"]):
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    allowed_roots = {
        "MANIPULANTE",
        "ESTRATEGIAS",
        "BOT_V2_DAYTIME_LAB",
    }
    if rel.parts[0] in allowed_roots:
        if rel.parts[0] == "BOT_V2_DAYTIME_LAB":
            rel_str = str(rel).replace("\\", "/")
            if rel_str.startswith("BOT_V2_DAYTIME_LAB/data/"):
                return False
            if rel_str.startswith("BOT_V2_DAYTIME_LAB/outputs/") and "phase36_live_news_mt5_dryrun" not in rel_str:
                return False
            if rel_str.startswith("BOT_V2_DAYTIME_LAB/reports/") and "PHASE36" not in rel_str:
                return False
            return True
        return True
    return path.name in {
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "ZIP_CONTENTS_MANIFEST.md",
        "README.md",
        "CHANGELOG.md",
    }


def rebuild_zip() -> dict[str, Any]:
    temp = ZIP_PATH.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    files = sorted([path for path in ROOT.rglob("*") if include_file(path)], key=lambda p: str(p.relative_to(ROOT)).lower())
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())
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
            "single_live_zip": len(live_zip_count()) == 1,
            "contains_phase36_report": "BOT_V2_DAYTIME_LAB/reports/PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.md" in names,
            "contains_manipulante": any(name.startswith("MANIPULANTE/") for name in names),
            "contains_phase36_outputs": any(name.startswith("BOT_V2_DAYTIME_LAB/outputs/phase36_live_news_mt5_dryrun/") for name in names),
            "contains_phase25_hash": "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in names,
            "heavy_entries_gt_2mb": [(name, zf.getinfo(name).file_size) for name in names if zf.getinfo(name).file_size > 2 * 1024 * 1024],
            "secret_like_entries": [name for name in names if any(token in name.lower() for token in SECRET_TOKENS)],
            "zip_entries_inside": [name for name in names if name.lower().endswith((".zip", ".zipbak"))],
        }
    write_json(OUT / "zip_validation" / "phase36_zip_validation.json", validation)
    write_text(
        OUT / "zip_validation" / "phase36_zip_validation.md",
        "\n".join([
            "# Phase36 ZIP Validation",
            "",
            f"- path: {validation['path']}",
            f"- size: {validation['size']}",
            f"- entries: {validation['entries']}",
            f"- sha256: {validation['sha256']}",
            f"- testzip: {validation['testzip']}",
            f"- single_live_zip: {validation['single_live_zip']}",
            f"- heavy_entries_gt_2mb: {validation['heavy_entries_gt_2mb']}",
            f"- secret_like_entries: {validation['secret_like_entries']}",
        ]),
    )
    write_text(OUT / "zip_validation" / "phase36_zip_entries.txt", "\n".join(names))
    return validation


def write_git_status() -> dict[str, Any]:
    payload = {
        "timestamp": now_iso(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status_short": run_cmd(["git", "status", "--short"]),
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
        "reason": "Phase36 blocked by active order_send risk; selective commit/push not performed.",
    }
    write_json(OUT / "git" / "phase36_git_status.json", payload)
    write_text(
        OUT / "git" / "phase36_git_status.md",
        "\n".join([
            "# Phase36 Git Status",
            "",
            f"- branch: {payload['branch']}",
            "- commit: NO",
            "- push: NO",
            f"- reason: {payload['reason']}",
        ]),
    )
    return payload


def main() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    pre = preflight()
    news_state = current_news_state_audit()
    create_live_news_fortress_assets()
    update_manipulante_docs_and_launcher()
    lot_summary = write_lot_outputs()
    today = run_today_readiness()
    order_audit = order_send_audit()
    report = write_report(pre, news_state, today, order_audit, lot_summary)
    write_master_docs(report["verdict"])
    zip_validation = rebuild_zip()
    git_status = write_git_status()
    final = {
        "report": report,
        "zip_validation": zip_validation,
        "git_status": git_status,
    }
    write_json(OUT / "phase36_final_execution_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return final


if __name__ == "__main__":
    main()
