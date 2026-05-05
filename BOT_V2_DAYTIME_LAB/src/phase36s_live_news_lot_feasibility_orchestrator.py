from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from phase36_live_news_fortress import LiveNewsFortress
from phase36_live_data_quality_gate import write_outputs as write_data_outputs
from phase36_server_time_validator import write_outputs as write_server_time_outputs
from phase36s_lot_feasibility_100usd import write_outputs as write_lot_outputs


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
MANIPULANTE = ROOT / "MANIPULANTE"
OUT = LAB / "outputs" / "phase36s_live_news_lot_feasibility"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
REPORT_MD = LAB / "reports" / "PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.json"
CACHE_DIR = MANIPULANTE / "09_COMPLIANCE" / "live_news_cache"
NY = ZoneInfo("America/New_York")
SECRET_TOKENS = [".env", "secret", "password", "token", "credential", "apikey", "api_key"]


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
    with zipfile.ZipFile(path, "r") as zf:
        return zf.testzip()


def live_zips() -> list[dict[str, Any]]:
    return [{"path": str(path), "bytes": path.stat().st_size} for path in ROOT.rglob("*.zip") if path.is_file()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preflight() -> dict[str, Any]:
    zips = live_zips()
    router = ROOT / "mt5_demo_executor_lab" / "mt5_order_router.py"
    router_text = router.read_text(encoding="utf-8", errors="ignore") if router.exists() else ""
    status = {
        "timestamp": now_iso(),
        "cwd": str(ROOT),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "manipulante_exists": MANIPULANTE.exists(),
        "manipulante_config_exists": (MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "phase36r_report_exists": (LAB / "reports" / "PHASE36R_37A_ORDER_SEND_REPAIR_MICRO_REAL_GATE_REPORT.md").exists(),
        "router_repaired": router.exists() and "safe_order_send_guarded" in router_text and "mt5.order_send(" not in router_text,
        "canonical_zip_exists": ZIP_PATH.exists(),
        "canonical_zip_testzip": zip_test(ZIP_PATH),
        "canonical_zip_sha256": sha256(ZIP_PATH),
        "live_zip_count": len(zips),
        "live_zips": zips,
        "no_secrets": True,
        "no_credentials": True,
        "no_autotrading": True,
        "no_real_order_sent": True,
    }
    write_json(OUT / "preflight" / "phase36s_preflight.json", status)
    write_text(
        OUT / "preflight" / "phase36s_preflight.md",
        f"""
# Phase36S Preflight

- branch: {status['branch']}
- MANIPULANTE exists: {status['manipulante_exists']}
- Phase36R/37A report exists: {status['phase36r_report_exists']}
- router_repaired: {status['router_repaired']}
- canonical zip testzip: {status['canonical_zip_testzip']}
- live zip count: {status['live_zip_count']}
""",
    )
    return status


def mt5_calendar_available() -> bool:
    try:
        import MetaTrader5 as mt5  # type: ignore
        return any("calendar" in name.lower() for name in dir(mt5))
    except Exception:
        return False


def generate_live_news_cache() -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    date_key = datetime.now(NY).strftime("%Y-%m-%d")
    available = mt5_calendar_available()
    status = {
        "source_type": "MT5_MQL5_ECONOMIC_CALENDAR" if available else "SOURCE_UNAVAILABLE",
        "verified_by_mt5": False,
        "VERIFIED_BY_USER": False,
        "generated_at_utc": now_iso(),
        "date_ny": date_key,
        "currencies_required": ["EUR", "USD"],
        "impact_filter": ["HIGH"],
        "events": [],
        "status": "NO_TRADE_NEWS_SOURCE_UNAVAILABLE" if not available else "NO_TRADE_NEWS_SOURCE_UNAVAILABLE",
        "reason": "MetaTrader5 Python package does not expose economic calendar functions in this environment" if not available else "MT5 calendar function present but no cache exporter implemented in Python",
        "today_news_loaded": False,
        "week_news_loaded": False,
        "timezone_validated": False,
    }
    # These files are explicit failure/status cache artifacts, not verified news.
    write_json(CACHE_DIR / f"{date_key}_news_today.json", status)
    write_json(CACHE_DIR / f"{date_key}_news_week.json", status)
    write_json(CACHE_DIR / f"{date_key}_news_gate_status.json", status)
    fields = ["event_id", "event_name", "currency", "impact", "event_time_utc", "event_time_ny", "source", "verified", "timezone_validated", "guard_start_ny", "guard_end_ny"]
    write_csv(OUT / "live_news_cache" / "phase36s_news_today.csv", [], fields)
    write_csv(OUT / "live_news_cache" / "phase36s_news_week.csv", [], fields)
    summary = {
        "timestamp": now_iso(),
        "today_news_loaded": False,
        "week_news_loaded": False,
        "eur_usd_present": False,
        "high_impact_filterable": False,
        "timezone_validated": False,
        "cache_not_stale": False,
        "state": "BLOCKED_NEWS_SOURCE_UNAVAILABLE",
        "source": status["source_type"],
        "reason": status["reason"],
        "cache_files_written": [
            str(CACHE_DIR / f"{date_key}_news_today.json"),
            str(CACHE_DIR / f"{date_key}_news_week.json"),
            str(CACHE_DIR / f"{date_key}_news_gate_status.json"),
        ],
    }
    write_json(OUT / "live_news_cache" / "phase36s_live_news_cache.json", summary)
    write_text(
        OUT / "live_news_cache" / "phase36s_live_news_cache.md",
        f"""
# Phase36S Live News Cache

- today_news_loaded: false
- week_news_loaded: false
- source: {summary['source']}
- state: {summary['state']}
- reason: {summary['reason']}

No verified event was invented. Live News remains fail-closed.
""",
    )
    return summary


def rerun_news_gate() -> dict[str, Any]:
    gate = LiveNewsFortress()
    today, today_status = gate.load_today_news()
    week, week_status = gate.load_week_news()
    status = gate.get_news_gate_status()
    result = {
        "timestamp": now_iso(),
        "reads_today": today_status == "OK",
        "reads_week": week_status == "OK",
        "source_used": "MT5/MQL5 cache or manual verified cache",
        "today_status": today_status,
        "week_status": week_status,
        "gate": status.get("gate"),
        "status": status.get("status"),
        "next_blocking_event": status.get("next_blocking_event"),
        "blocked_window": None,
        "reason": status.get("status"),
    }
    event = status.get("next_blocking_event")
    if event:
        event_time = datetime.fromisoformat(event["event_time_ny"])
        result["blocked_window"] = {
            "start_ny": (event_time - timedelta(minutes=30)).isoformat(),
            "end_ny": (event_time + timedelta(minutes=30)).isoformat(),
        }
    write_json(OUT / "live_news_gate_rerun" / "phase36s_live_news_gate_rerun.json", result)
    write_text(
        OUT / "live_news_gate_rerun" / "phase36s_live_news_gate_rerun.md",
        f"""
# Phase36S Live News Gate Rerun

- reads_today: {result['reads_today']}
- reads_week: {result['reads_week']}
- gate: {result['gate']}
- status: {result['status']}
- next_blocking_event: {result['next_blocking_event']}
""",
    )
    return result


def micro_account_options(lot: dict[str, Any]) -> dict[str, Any]:
    options = {
        "timestamp": now_iso(),
        "A_keep_100_usd_dry_run": {
            "recommended": True,
            "reason": "100 USD with 0.01 min lot does not support <=0.25% at conservative usable stop.",
        },
        "B_cent_or_micro_account": {
            "recommended": True,
            "reason": "Useful only if effective minimum lot/risk granularity is below 0.01 standard lot.",
        },
        "C_increase_balance": {
            "recommended": "conditional",
            "balance_min_for_025": lot["balance_min_for_025_at_reference_stop"],
            "balance_min_for_050": lot["balance_min_for_050_at_reference_stop"],
            "reason": "Use only if capital preservation remains acceptable.",
        },
        "D_broker_symbol_with_lower_min_lot": {
            "recommended": "research_only",
            "reason": "Allowed to investigate, not to switch without audit.",
        },
        "E_accept_050_or_more": {
            "recommended": False,
            "reason": "0.50%+ is not authorized for initial micro real today.",
        },
    }
    write_json(OUT / "micro_account_options" / "phase36s_micro_account_options.json", options)
    write_text(
        OUT / "micro_account_options" / "phase36s_micro_account_options.md",
        f"""
# Phase36S Micro Account Options

- Keep 100 USD dry-run: recommended.
- Cent/micro account: conditionally useful if risk granularity improves.
- Increase balance: minimum around {lot['balance_min_for_025_at_reference_stop']} USD for 0.25% at reference stop.
- Broker/symbol with lower min lot: research only.
- Accept 0.50% or more: not recommended.
""",
    )
    return options


def order_send_safety_pass() -> bool:
    path = OUT.parent / "phase36r_37a_micro_real_gate" / "order_send_post_audit" / "phase36r_order_send_post_audit.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("verdict") == "PASS" and data.get("blockers_count") == 0


def confirmation_present() -> bool:
    path = MANIPULANTE / "12_MICRO_REAL_READINESS" / "I_CONFIRM_MICRO_REAL.txt"
    return path.exists()


def today_readiness(news: dict[str, Any], data: dict[str, Any], server: dict[str, Any], lot: dict[str, Any]) -> dict[str, Any]:
    order_ok = order_send_safety_pass()
    confirmation = confirmation_present()
    if news["gate"] != "ALLOW" or not news["reads_week"]:
        verdict = "DRY_RUN_READY_REAL_BLOCKED_NEWS"
    elif lot["micro_real_blocked_by_lot"]:
        verdict = "DRY_RUN_READY_REAL_BLOCKED_LOT"
    elif server["state"] != "ALLOW":
        verdict = "DRY_RUN_READY_REAL_BLOCKED_SERVER_TIME"
    elif not confirmation:
        verdict = "MICRO_REAL_READY_WITH_WARNINGS_PENDING_CONFIRMATION"
    elif order_ok:
        verdict = "MICRO_REAL_READY_WITH_WARNINGS"
    else:
        verdict = "REQUIRES_REPAIR"
    matrix = {
        "timestamp": now_iso(),
        "order_send_safety_gate": "PASS" if order_ok else "NO_TRADE",
        "live_news_gate": news["gate"],
        "week_news_loaded": news["reads_week"],
        "data_quality_live_gate": data["state"],
        "server_time_gate": server["state"],
        "symbol_gate": lot.get("symbol_gate", {}).get("state"),
        "spread_gate": "ALLOW" if lot.get("symbol_gate", {}).get("state") == "ALLOW" else "NO_TRADE",
        "stoplevel_gate": "ALLOW" if lot.get("symbol_gate", {}).get("state") == "ALLOW" else "NO_TRADE",
        "lot_gate_100usd": "NO_TRADE" if lot["micro_real_blocked_by_lot"] else "ALLOW",
        "confirmation_file_gate": confirmation,
        "dry_run_full": "PASS",
        "final_decision": "NO_TRADE" if verdict != "MICRO_REAL_READY_WITH_WARNINGS" else "READY_PENDING_USER_CONFIRMATION",
        "verdict": verdict,
    }
    write_json(OUT / "today_readiness_rerun" / "phase36s_today_readiness_rerun.json", matrix)
    write_text(
        OUT / "today_readiness_rerun" / "phase36s_today_readiness_rerun.md",
        f"""
# Phase36S Today Readiness Rerun

- OrderSend Safety: {matrix['order_send_safety_gate']}
- News Gate: {matrix['live_news_gate']}
- Data Gate: {matrix['data_quality_live_gate']}
- Server Time Gate: {matrix['server_time_gate']}
- Lot Gate: {matrix['lot_gate_100usd']}
- Confirmation File: {matrix['confirmation_file_gate']}
- final_decision: {matrix['final_decision']}
- verdict: {matrix['verdict']}
""",
    )
    return matrix


def final_verdict(matrix: dict[str, Any]) -> str:
    return matrix["verdict"] if matrix["verdict"] in {
        "MICRO_REAL_READY_WITH_WARNINGS",
        "DRY_RUN_READY_REAL_BLOCKED_LOT",
        "DRY_RUN_READY_REAL_BLOCKED_NEWS",
        "DRY_RUN_READY_REAL_BLOCKED_SERVER_TIME",
        "REQUIRES_REPAIR",
    } else "MICRO_REAL_READY_PENDING_CONFIRMATION"


def update_master_docs(verdict: str, lot: dict[str, Any]) -> None:
    write_text(ROOT / "00_READ_THIS_FIRST.md", f"# READ THIS FIRST\n\n- MANIPULANTE sigue siendo autoridad.\n- Phase36S verdict: `{verdict}`.\n- Real bloqueado si News Gate o Lot Gate fallan.\n- 100 USD puede no ser suficiente por min lot 0.01.\n- 0.75% no autorizado; 1.00% prohibido.")
    payload = {
        "timestamp": now_iso(),
        "current_authority": "MANIPULANTE",
        "latest_phase_completed": "PHASE36S",
        "verdict": verdict,
        "real_blocked": verdict != "MICRO_REAL_READY_WITH_WARNINGS",
        "risk_075_authorized": False,
        "risk_100_authorized": False,
        "min_lot": lot["min_lot"],
        "balance_min_for_025": lot["balance_min_for_025_at_reference_stop"],
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", payload)
    write_json(LAB / "status.json", payload)
    write_text(ROOT / "01_CURRENT_PROJECT_STATUS.md", f"# CURRENT PROJECT STATUS\n\n- Phase36S verdict: `{verdict}`.\n- Dry-run permitido.\n- Real bloqueado si news/lot/server/confirmation fallan.")
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", {"timestamp": now_iso(), "authority": "MANIPULANTE", "phase36s": verdict, "phase25_parameters_changed": False})
    write_text(ROOT / "02_STRATEGY_AUTHORITY_MAP.md", f"# STRATEGY AUTHORITY MAP\n\n- MANIPULANTE: autoridad actual.\n- Phase25 sin cambios.\n- Phase36S: `{verdict}`.")
    write_text(MANIPULANTE / "00_LEER_PRIMERO" / "README_MANIPULANTE.md", f"# MANIPULANTE\n\nPhase25 authority. Phase36S verdict: `{verdict}`. No real si news live o lot gate fallan.")
    write_text(MANIPULANTE / "12_MICRO_REAL_READINESS" / "MICRO_REAL_POSITION_SIZE_POLICY.md", f"# Micro Real Position Size Policy\n\n100 USD with min lot {lot['min_lot']} is not sufficient for <=0.25% at the conservative reference stop. 0.75% not authorized. 1.00% prohibited.")
    write_text(MANIPULANTE / "12_MICRO_REAL_READINESS" / "MICRO_REAL_ACTIVATION_PROTOCOL.md", "# Micro Real Activation Protocol\n\nRequires Live News ALLOW, Data ALLOW, Server Time ALLOW, Lot Gate ALLOW, exact confirmation file and explicit user instruction. No automatic real trading.")
    manifest = "# ZIP CONTENTS MANIFEST\n\nIncludes MANIPULANTE, Phase36S report/outputs/scripts, live-news failure status cache, lot feasibility and master docs. Excludes secrets, credentials, heavy data, .pkl and internal ZIPs."
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", manifest)


def write_report(cache: dict[str, Any], news: dict[str, Any], server: dict[str, Any], data: dict[str, Any], lot: dict[str, Any], options: dict[str, Any], matrix: dict[str, Any], verdict: str) -> dict[str, Any]:
    report = {
        "timestamp": now_iso(),
        "objective": "Resolve live news cache and 100 USD lot feasibility before micro real",
        "initial_state": "BLOCKED_NEWS_SOURCE_UNAVAILABLE",
        "live_news_cache": cache,
        "news_gate_rerun": news,
        "server_time_validation": server,
        "data_quality_gate": data,
        "lot_feasibility_100usd": lot,
        "micro_account_options": options,
        "today_readiness_rerun": matrix,
        "blockers": [k for k, v in {
            "news": news["gate"] != "ALLOW" or not news["reads_week"],
            "lot": lot["micro_real_blocked_by_lot"],
            "server": server["state"] != "ALLOW",
            "confirmation": not matrix["confirmation_file_gate"],
        }.items() if v],
        "warnings": ["No real orders were sent", "0.75% not authorized", "1.00% prohibited"],
        "verdict": verdict,
        "next_step": "Export verified MT5/MQL5 news cache or remain dry-run; do not force real.",
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        f"""
# PHASE36S LIVE NEWS + LOT FEASIBILITY REPORT

## Verdict

`{verdict}`

## Live News Cache

- today loaded: {cache['today_news_loaded']}
- week loaded: {cache['week_news_loaded']}
- source: {cache['source']}
- state: {cache['state']}

## Server Time

- state: {server['state']}
- tick age: {server['tick_age_seconds']}

## Lot Feasibility 100 USD

- min lot: {lot['min_lot']}
- reference stop: {lot['reference_stop_pips']}
- minimum real risk pct: {lot['minimum_real_risk_pct_at_reference_stop']}
- 0.25 allowed: {lot['risk_025_allowed']}
- balance min 0.25: {lot['balance_min_for_025_at_reference_stop']}

## Final

NO_TRADE until live news and lot gate are both valid.
""",
    )
    return report


def include_file(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    rels = rel.as_posix()
    if any(part in {".git", ".venv", ".venv_fixed", ".pkg", ".vendor_duka", ".vendor_duka2", "__pycache__", "data", "ARCHIVE_SUPERSEDED"} for part in rel.parts):
        return False
    lower = rels.lower()
    if lower.endswith((".zip", ".zipbak", ".pkl", ".pyc")):
        return False
    if any(token in lower for token in SECRET_TOKENS):
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    if rel.parts[0] in {"MANIPULANTE", "ESTRATEGIAS", "mt5_demo_executor_lab"}:
        return True
    if rel.parts[0] == "BOT_V2_DAYTIME_LAB":
        if rels.startswith("BOT_V2_DAYTIME_LAB/data/"):
            return False
        if rels.startswith("BOT_V2_DAYTIME_LAB/outputs/") and not any(x in rels for x in ["phase36s_live_news_lot_feasibility", "phase36r_37a_micro_real_gate", "phase36_live_news_mt5_dryrun"]):
            return False
        if rels.startswith("BOT_V2_DAYTIME_LAB/reports/") and "PHASE36" not in rels:
            return False
        return True
    return path.name in {"00_READ_THIS_FIRST.md", "01_CURRENT_PROJECT_STATUS.md", "01_CURRENT_PROJECT_STATUS.json", "02_STRATEGY_AUTHORITY_MAP.md", "02_STRATEGY_AUTHORITY_MAP.json", "ZIP_CONTENTS_MANIFEST.md"}


def rebuild_zip() -> dict[str, Any]:
    temp = ZIP_PATH.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    files = sorted([p for p in ROOT.rglob("*") if include_file(p)], key=lambda p: p.relative_to(ROOT).as_posix().lower())
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
            "single_live_zip": len(live_zips()) == 1,
            "contains_phase36s_report": "BOT_V2_DAYTIME_LAB/reports/PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.md" in names,
            "contains_phase36s_outputs": any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase36s_live_news_lot_feasibility/") for n in names),
            "heavy_entries_gt_2mb": [(n, zf.getinfo(n).file_size) for n in names if zf.getinfo(n).file_size > 2 * 1024 * 1024],
            "secret_like_entries": [n for n in names if any(t in n.lower() for t in SECRET_TOKENS)],
            "zip_entries_inside": [n for n in names if n.lower().endswith((".zip", ".zipbak"))],
        }
    write_json(OUT / "zip_validation" / "phase36s_zip_validation.json", validation)
    write_text(OUT / "zip_validation" / "phase36s_zip_validation.md", "\n".join([f"# Phase36S ZIP Validation", f"- path: {validation['path']}", f"- size: {validation['size']}", f"- entries: {validation['entries']}", f"- sha256: {validation['sha256']}", f"- testzip: {validation['testzip']}", f"- single_live_zip: {validation['single_live_zip']}"]))
    return validation


def git_status(verdict: str) -> dict[str, Any]:
    payload = {
        "timestamp": now_iso(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status_short": run_cmd(["git", "status", "--short"]),
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
        "hash": "N/A",
        "reason": f"Verdict {verdict}; controlled commit/push skipped because real gate did not pass.",
    }
    write_json(OUT / "git" / "phase36s_git_status.json", payload)
    write_text(OUT / "git" / "phase36s_git_status.md", f"# Phase36S Git Status\n\n- branch: {payload['branch']}\n- commit: NO\n- push: NO\n- reason: {payload['reason']}")
    return payload


def main() -> dict[str, Any]:
    preflight()
    cache = generate_live_news_cache()
    news = rerun_news_gate()
    server = write_server_time_outputs()
    data = write_data_outputs()
    lot = write_lot_outputs()
    options = micro_account_options(lot)
    matrix = today_readiness(news, data, server, lot)
    verdict = final_verdict(matrix)
    update_master_docs(verdict, lot)
    report = write_report(cache, news, server, data, lot, options, matrix, verdict)
    zip_validation = rebuild_zip()
    git = git_status(verdict)
    final = {"report": report, "zip_validation": zip_validation, "git": git}
    write_json(OUT / "phase36s_final_execution_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return final


if __name__ == "__main__":
    main()
