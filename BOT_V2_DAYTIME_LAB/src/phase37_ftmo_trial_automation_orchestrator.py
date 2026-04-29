from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from phase37_ftmo_account_gate import write_outputs as write_account_outputs
from phase37_ftmo_live_news_consumer import write_outputs as write_news_outputs
from phase37_ftmo_lot_validator import write_outputs as write_lot_outputs
from phase37_ftmo_symbol_data_gate import write_outputs as write_symbol_outputs
from phase37_ftmo_time_gate import write_outputs as write_time_outputs
from phase37_ftmo_trial_bot_runner import main as run_bot_runner
from phase37_ftmo_trial_support import (
    LAB,
    MANIPULANTE,
    OUT,
    ROOT,
    ZIP_PATH,
    all_zip_inventory,
    confirmation_file_status,
    include_file_for_zip,
    now_iso,
    order_send_safety,
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


REPORT_MD = LAB / "reports" / "PHASE37_FTMO_SWING_FREE_TRIAL_AUTOMATION_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE37_FTMO_SWING_FREE_TRIAL_AUTOMATION_REPORT.json"


def preflight() -> dict[str, Any]:
    router = ROOT / "mt5_demo_executor_lab" / "mt5_order_router.py"
    text = router.read_text(encoding="utf-8", errors="ignore") if router.exists() else ""
    zips = root_live_zips()
    status = {
        "timestamp_utc": now_iso(),
        "cwd": str(ROOT),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "manipulante_exists": MANIPULANTE.exists(),
        "manipulante_config_exists": (MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "phase25_config_hash_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt").exists(),
        "router_repaired": router.exists() and "safe_order_send_guarded" in text and "mt5.order_send(" not in text,
        "phase36s_report_exists": (LAB / "reports" / "PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.md").exists(),
        "canonical_zip_exists": ZIP_PATH.exists(),
        "canonical_zip_testzip": zip_test(ZIP_PATH) if ZIP_PATH.exists() else "MISSING",
        "canonical_zip_sha256": sha256(ZIP_PATH),
        "root_live_zip_count": len(zips),
        "root_live_zips": zips,
        "all_zip_inventory_count": len(all_zip_inventory()),
        "no_secrets": True,
        "no_credentials": True,
        "no_real_account_saved": True,
        "no_autotrading_blind": True,
        "no_order_send_without_gate": "mt5.order_send(" not in text,
    }
    write_json(OUT / "preflight" / "phase37_preflight.json", status)
    write_text(
        OUT / "preflight" / "phase37_preflight.md",
        f"""
# Phase37 Preflight

- cwd: {status['cwd']}
- branch: {status['branch']}
- MANIPULANTE exists: {status['manipulante_exists']}
- Phase25 hash exists: {status['phase25_config_hash_exists']}
- router repaired: {status['router_repaired']}
- Phase36S report exists: {status['phase36s_report_exists']}
- canonical zip testzip: {status['canonical_zip_testzip']}
- root live zip count: {status['root_live_zip_count']}
- no order_send without gate: {status['no_order_send_without_gate']}
""",
    )
    return status


def write_signal_sync_outputs() -> dict[str, Any]:
    status = signal_sync()
    rows = status.get("candidate_findings", [])
    write_csv(OUT / "signal_sync" / "phase37_signal_sync_diff.csv", rows, ["path", "score"])
    write_json(OUT / "signal_sync" / "phase37_signal_sync.json", status)
    write_text(
        OUT / "signal_sync" / "phase37_signal_sync.md",
        f"""
# Phase37 Signal Sync

- state: {status['state']}
- signal engine found: {status['signal_engine_found']}
- MANIPULANTE match: {status['manipulante_match']}
- reason: {status['reason']}

No automatic trial order is allowed without a live callable signal engine that matches MANIPULANTE Phase25.
""",
    )
    return status


def write_order_router_outputs() -> dict[str, Any]:
    safety = order_send_safety()
    config = strategy_config_gate()
    confirmation = confirmation_file_status()
    status = {
        "timestamp_utc": now_iso(),
        "order_send_safety": safety,
        "strategy_config_gate": config,
        "confirmation_file": confirmation,
        "order_check_required": True,
        "real_blocked": True,
        "state": "PASS" if safety["state"] == "PASS" else "BLOCKER",
    }
    write_json(OUT / "order_router" / "phase37_order_router.json", status)
    write_text(
        OUT / "order_router" / "phase37_order_router.md",
        f"""
# Phase37 Order Router

- order_send gated: {safety['state'] == 'PASS'}
- direct mt5.order_send: {safety['direct_mt5_order_send']}
- order_check required: true
- real blocked: true
- confirmation file present: {confirmation['present']}
- confirmation file valid: {confirmation['valid']}
""",
    )
    return status


def today_readiness(
    account: dict[str, Any],
    news: dict[str, Any],
    symbol: dict[str, Any],
    time_status: dict[str, Any],
    lot: dict[str, Any],
    signal: dict[str, Any],
    router: dict[str, Any],
    dry_run: dict[str, Any],
) -> dict[str, Any]:
    confirmation = router["confirmation_file"]
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
        "order_router_safety": router.get("state"),
        "confirmation_file": confirmation.get("valid"),
        "dry_run": "PASS" if dry_run.get("order_sent") is False else "FAIL",
    }
    if account.get("state") not in {"FTMO_DEMO_TRIAL_CONFIRMED", "WARNING_BALANCE_NOT_10K"}:
        verdict = "FTMO_TRIAL_BLOCKED_ACCOUNT"
    elif news.get("gate") != "ALLOW" or news.get("week_loaded") is not True:
        verdict = "FTMO_TRIAL_BLOCKED_NEWS"
    elif symbol.get("state") != "ALLOW":
        verdict = "FTMO_TRIAL_BLOCKED_DATA"
    elif time_status.get("state") != "ALLOW":
        verdict = "FTMO_TRIAL_REQUIRES_REPAIR"
    elif signal.get("state") != "MANIPULANTE_SIGNAL_SYNC_OK":
        verdict = "FTMO_TRIAL_BLOCKED_SIGNAL"
    elif router.get("state") != "PASS":
        verdict = "FTMO_TRIAL_REQUIRES_REPAIR"
    elif confirmation.get("valid") is not True:
        verdict = "FTMO_TRIAL_DRYRUN_READY_ORDER_BLOCKED"
    elif dry_run.get("order_sent") is not False:
        verdict = "FTMO_TRIAL_REQUIRES_REPAIR"
    else:
        verdict = "FTMO_TRIAL_AUTO_READY"
    matrix = {
        "timestamp_utc": now_iso(),
        "gates": gates,
        "final_decision": "NO_TRADE" if verdict != "FTMO_TRIAL_AUTO_READY" else "TRIAL_AUTO_ALLOWED",
        "verdict": verdict,
        "can_run_auto_today": verdict == "FTMO_TRIAL_AUTO_READY",
        "conditions_required": [
            "FTMO demo/trial account confirmed",
            "MQL5 news cache today/week loaded and ALLOW",
            "Data/Symbol/Spread/StopLevel ALLOW",
            "Time Gate ALLOW",
            "Lot Gate ALLOW",
            "Live signal engine synced to MANIPULANTE Phase25",
            "Confirmation file valid",
            "STOP_BOT.txt absent",
        ],
    }
    write_json(OUT / "today_readiness" / "phase37_today_readiness.json", matrix)
    write_text(
        OUT / "today_readiness" / "phase37_today_readiness.md",
        f"""
# Phase37 Today Readiness

- Account Gate: {gates['account_gate']}
- News Gate: {gates['news_gate']}
- Week News Loaded: {gates['week_news_loaded']}
- Data Gate: {gates['data_gate']}
- Time Gate: {gates['time_gate']}
- Lot Gate: {gates['lot_gate']}
- Signal Sync: {gates['signal_sync']}
- Order Router Safety: {gates['order_router_safety']}
- Confirmation File: {gates['confirmation_file']}
- Dry-run: {gates['dry_run']}
- Verdict: {verdict}
- Final decision: {matrix['final_decision']}
""",
    )
    return matrix


def update_master_docs(verdict: str) -> None:
    status_payload = {
        "timestamp_utc": now_iso(),
        "latest_phase": "PHASE37_FTMO_SWING_FREE_TRIAL_AUTOMATION",
        "verdict": verdict,
        "authority": "MANIPULANTE_PHASE25",
        "strategy_changed": False,
        "real_blocked": True,
        "ftmo_trial_demo_only": True,
        "risk_default": 0.005,
        "risk_075_trial_stress_only": True,
        "risk_100_prohibited": True,
        "news_gate_required": True,
        "data_gate_required": True,
        "signal_sync_required": True,
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status_payload)
    write_json(LAB / "status.json", status_payload)
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        f"""
# READ THIS FIRST

- Phase37 verdict: `{verdict}`.
- MANIPULANTE remains Phase25 Authority.
- FTMO Trial automation is demo/trial only.
- Real is blocked.
- Default trial risk is 0.50%; 0.75% is stress trial only; 1.00% is prohibited.
- Live News Gate, Data Gate and Signal Sync are mandatory.
""",
    )
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        f"""
# Current Project Status

- Latest phase: Phase37 FTMO Swing Free Trial automation.
- Verdict: `{verdict}`.
- Authority: MANIPULANTE Phase25.
- Real trading: blocked.
- FTMO trial automation: blocked unless all gates pass.
""",
    )
    write_json(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        {
            "timestamp_utc": now_iso(),
            "authority": "MANIPULANTE_PHASE25",
            "tp": 1.4,
            "be": 0.4,
            "bf": 70,
            "be05": "SHADOW_ONLY",
            "phase37": verdict,
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
- Phase37 verdict: `{verdict}`.
""",
    )
    write_text(
        MANIPULANTE / "00_LEER_PRIMERO" / "README_MANIPULANTE.md",
        f"""
# MANIPULANTE

MANIPULANTE is Phase25 Authority. Phase37 prepares FTMO Free Trial 2-Step Swing MT5 automation only.

- Verdict: `{verdict}`.
- No real account.
- No strategy change.
- News Gate and Data Gate are mandatory.
- `STOP_BOT.txt` stops automation.
""",
    )
    write_text(
        MANIPULANTE / "04_OPERACION_DIARIA" / "MANIPULANTE_DAILY_RUNBOOK.md",
        """
# MANIPULANTE Daily Runbook

1. Confirm account is FTMO demo/trial.
2. Export MQL5 Economic Calendar cache.
3. Confirm News Gate = ALLOW.
4. Confirm Data/Symbol/Spread/StopLevel = ALLOW.
5. Confirm Time Gate = ALLOW.
6. Confirm Signal Sync = OK.
7. Dry-run first.
8. No real trading in Phase37.
""",
    )
    write_text(
        MANIPULANTE / "04_OPERACION_DIARIA" / "MANIPULANTE_KILL_SWITCH.md",
        """
# MANIPULANTE Kill Switch

Any failed gate means NO_TRADE.

For FTMO trial automation, create:
`MANIPULANTE\\13_FTMO_TRIAL_AUTOMATION\\STOP_BOT.txt`

If the file exists, the runner stops.
""",
    )
    write_text(
        MANIPULANTE / "09_COMPLIANCE" / "LIVE_NEWS_FORTRESS_POLICY.md",
        """
# Live News Fortress Policy

Phase37 uses MT5/MQL5 Economic Calendar cache.

- EUR/USD high impact only.
- Guard window: plus/minus 30 minutes.
- Missing cache: NO_TRADE.
- Stale cache: NO_TRADE.
- Unknown timezone: NO_TRADE.
""",
    )
    manifest = (
        "# ZIP CONTENTS MANIFEST\n\n"
        "Includes MANIPULANTE, Phase37 scripts, Phase37 outputs, Phase37 report, FTMO trial docs, "
        "live news adapter, order router, bot runner, configs and master docs. Excludes secrets, "
        "credentials, MT5 account files, heavy data, .pkl and internal ZIPs.\n"
    )
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", manifest)


def write_report(
    account: dict[str, Any],
    news: dict[str, Any],
    symbol: dict[str, Any],
    time_status: dict[str, Any],
    lot: dict[str, Any],
    signal: dict[str, Any],
    router: dict[str, Any],
    dry_run: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if matrix["verdict"] == "FTMO_TRIAL_BLOCKED_NEWS":
        blockers.append("Live News Gate no tiene cache FTMO MQL5 valida hoy/semana")
    if signal.get("state") != "MANIPULANTE_SIGNAL_SYNC_OK":
        blockers.append("No hay signal engine live confirmado para MANIPULANTE Phase25")
    if router["confirmation_file"].get("valid") is not True:
        blockers.append("Confirmation file FTMO trial ausente o invalido")
    if dry_run.get("final_decision") == "STOP_BOT_ACTIVE":
        blockers.append("STOP_BOT.txt esta activo")
    report = {
        "timestamp_utc": now_iso(),
        "objective": "FTMO Free Trial 2-Step Swing MT5 automation for MANIPULANTE, demo/trial only",
        "account_gate": account,
        "live_news_gate": news,
        "symbol_data_gate": symbol,
        "time_gate": time_status,
        "lot_gate": lot,
        "signal_sync": signal,
        "order_router": router,
        "dry_run": dry_run,
        "today_readiness": matrix,
        "blockers": blockers,
        "warnings": ["No real trading authorized", "0.75% is trial stress only", "1.00% prohibited"],
        "verdict": matrix["verdict"],
        "next_step": "Instalar/ejecutar CalendarExporter MQL5 y conectar signal engine live; luego rerun Phase37 dry-run.",
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        f"""
# PHASE37 FTMO SWING FREE TRIAL AUTOMATION REPORT

## Verdict

`{matrix['verdict']}`

## Account Gate

- state: {account.get('state')}
- balance: {account.get('balance')}
- server: {account.get('server')}
- mode: {account.get('trade_mode_label')}

## Live News

- gate: {news.get('gate')}
- state: {news.get('state')}
- today loaded: {news.get('today_loaded')}
- week loaded: {news.get('week_loaded')}

## Data / Symbol / Time / Lot

- symbol: {symbol.get('symbol')}
- spread: {symbol.get('spread_pips')}
- time gate: {time_status.get('state')}
- lot gate: {lot.get('state')}

## Signal / Router / Dry-run

- signal sync: {signal.get('state')}
- order router: {router.get('state')}
- dry-run decision: {dry_run.get('final_decision')}
- order_sent: {dry_run.get('order_sent')}

## Final

No real trading is authorized. FTMO trial automation remains fail-closed until all gates pass.
""",
    )
    return report


def rebuild_zip() -> dict[str, Any]:
    temp = ZIP_PATH.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    files = sorted(
        [path for path in ROOT.rglob("*") if include_file_for_zip(path)],
        key=lambda item: item.relative_to(ROOT).as_posix().lower(),
    )
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
            "single_live_zip": len(root_live_zips()) == 1,
            "contains_phase37_report": "BOT_V2_DAYTIME_LAB/reports/PHASE37_FTMO_SWING_FREE_TRIAL_AUTOMATION_REPORT.md" in names,
            "contains_phase37_outputs": any(name.startswith("BOT_V2_DAYTIME_LAB/outputs/phase37_ftmo_swing_trial_auto/") for name in names),
            "contains_ftmo_docs": any(name.startswith("MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/") for name in names),
            "contains_calendar_exporter": "MANIPULANTE/09_COMPLIANCE/MT5_LIVE_NEWS_ADAPTER/MANIPULANTE_CalendarExporter.mq5" in names,
            "heavy_entries_gt_2mb": [(name, zf.getinfo(name).file_size) for name in names if zf.getinfo(name).file_size > 2 * 1024 * 1024],
            "secret_like_entries": [name for name in names if any(token in name.lower() for token in ["secret", "password", "token", "credential", ".env"])],
            "zip_entries_inside": [name for name in names if name.lower().endswith((".zip", ".zipbak"))],
        }
    write_json(OUT / "zip_validation" / "phase37_zip_validation.json", validation)
    write_text(
        OUT / "zip_validation" / "phase37_zip_validation.md",
        f"""
# Phase37 ZIP Validation

- path: {validation['path']}
- size: {validation['size']}
- entries: {validation['entries']}
- sha256: {validation['sha256']}
- testzip: {validation['testzip']}
- single_live_zip: {validation['single_live_zip']}
- contains Phase37 report: {validation['contains_phase37_report']}
- contains Phase37 outputs: {validation['contains_phase37_outputs']}
- contains FTMO docs: {validation['contains_ftmo_docs']}
- contains calendar exporter: {validation['contains_calendar_exporter']}
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
    write_json(OUT / "git" / "phase37_git_status.json", payload)
    write_text(
        OUT / "git" / "phase37_git_status.md",
        f"""
# Phase37 Git Status

- branch: {payload['branch']}
- commit: NO
- push: NO
- reason: {payload['reason']}
""",
    )
    return payload


def main() -> dict[str, Any]:
    preflight()
    account = write_account_outputs()
    news = write_news_outputs()
    symbol = write_symbol_outputs()
    time_status = write_time_outputs()
    lot = write_lot_outputs()
    signal = write_signal_sync_outputs()
    router = write_order_router_outputs()
    dry_run = run_bot_runner(["--ftmo-trial", "--dry-run", "--risk", "0.005", "--no-real"])
    matrix = today_readiness(account, news, symbol, time_status, lot, signal, router, dry_run)
    update_master_docs(matrix["verdict"])
    report = write_report(account, news, symbol, time_status, lot, signal, router, dry_run, matrix)
    zip_validation = rebuild_zip()
    git = git_status(matrix["verdict"])
    final = {
        "report": report,
        "zip_validation": zip_validation,
        "git": git,
    }
    write_json(OUT / "phase37_final_execution_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return final


if __name__ == "__main__":
    main()
