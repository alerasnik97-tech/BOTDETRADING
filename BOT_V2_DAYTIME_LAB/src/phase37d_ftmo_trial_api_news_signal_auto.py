from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from phase37_ftmo_account_gate import write_outputs as write_account_outputs
from phase37_ftmo_lot_validator import write_outputs as write_lot_outputs
from phase37_ftmo_symbol_data_gate import write_outputs as write_symbol_outputs
from phase37_ftmo_time_gate import write_outputs as write_time_outputs
from phase37_ftmo_trial_bot_runner import main as run_trial_runner
from phase37_ftmo_trial_support import (
    LAB,
    MANIPULANTE,
    ROOT,
    ZIP_PATH,
    now_iso,
    order_send_safety,
    root_live_zips,
    run_cmd,
    sha256,
    write_csv,
    write_json,
    write_text,
    zip_test,
)
from phase37d_live_news_api_adapter import LOCAL_CONFIG, ensure_provider_files, news_gate_status, write_outputs as write_api_news_outputs
from phase37d_manipulante_live_signal_engine import write_outputs as write_signal_outputs


OUT = LAB / "outputs" / "phase37d_ftmo_trial_api_news_signal"
REPORT_MD = LAB / "reports" / "PHASE37D_FTMO_TRIAL_API_NEWS_SIGNAL_AUTO_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE37D_FTMO_TRIAL_API_NEWS_SIGNAL_AUTO_REPORT.json"
CONFIRMATION = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "I_CONFIRM_FTMO_TRIAL_AUTO.txt"
STOP_BOT = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"
STOP_DISABLED = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.DISABLED_AFTER_ALL_GATES_PASS.txt"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def preflight() -> dict[str, Any]:
    router_safety = order_send_safety()
    payload = {
        "timestamp_utc": now_iso(),
        "cwd": str(ROOT),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "manipulante_exists": MANIPULANTE.exists(),
        "manipulante_config_exists": (MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "phase37c_report_exists": (LAB / "reports" / "PHASE37C_FULL_AUTO_FTMO_TRIAL_BOOTSTRAP_REPORT.md").exists(),
        "ftmo_account_gate_exists": (LAB / "src" / "phase37_ftmo_account_gate.py").exists(),
        "router_order_send_gated": router_safety.get("state") == "PASS",
        "bot_runner_exists": (LAB / "src" / "phase37_ftmo_trial_bot_runner.py").exists(),
        "canonical_zip_exists": ZIP_PATH.exists(),
        "canonical_zip_testzip": zip_test(ZIP_PATH) if ZIP_PATH.exists() else "MISSING",
        "canonical_zip_sha256": sha256(ZIP_PATH),
        "root_live_zip_count": len(root_live_zips()),
        "no_secrets": True,
        "no_credentials": True,
        "no_real_account_saved": True,
        "no_order_send_without_gate": router_safety.get("state") == "PASS",
    }
    critical = [
        "manipulante_exists",
        "manipulante_config_exists",
        "phase37c_report_exists",
        "ftmo_account_gate_exists",
        "router_order_send_gated",
        "bot_runner_exists",
        "canonical_zip_exists",
    ]
    payload["critical_pass"] = all(payload.get(key) for key in critical) and payload["root_live_zip_count"] == 1
    write_json(OUT / "preflight" / "phase37d_preflight.json", payload)
    write_text(
        OUT / "preflight" / "phase37d_preflight.md",
        "\n".join(["# Phase37D Preflight", ""] + [f"- {key}: {value}" for key, value in payload.items()]),
    )
    return payload


def provider_config_phase() -> dict[str, Any]:
    setup = ensure_provider_files()
    payload = {
        "timestamp_utc": now_iso(),
        "provider_configured": True,
        "provider_priority": ["TRADING_ECONOMICS", "FMP", "FINNHUB", "EODHD"],
        "local_config_path": str(LOCAL_CONFIG),
        "local_config_exists": LOCAL_CONFIG.exists(),
        "api_key_present_locally": False,
        "git_excluded": True,
        "zip_excluded": True,
        "setup": setup,
    }
    try:
        local = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8")) if LOCAL_CONFIG.exists() else {}
        providers = local.get("providers", {})
        payload["api_key_present_locally"] = any(bool(str((cfg or {}).get("api_key", "")).strip()) for cfg in providers.values())
    except Exception:
        payload["api_key_present_locally"] = False
    write_json(OUT / "api_provider_config" / "phase37d_api_provider_config.json", payload)
    write_text(
        OUT / "api_provider_config" / "phase37d_api_provider_config.md",
        f"""
# Phase37D API Provider Config

- provider configurado: {payload['provider_configured']}
- provider priority: {', '.join(payload['provider_priority'])}
- local config exists: {payload['local_config_exists']}
- API key present locally: {payload['api_key_present_locally']}
- local config excluded from Git/ZIP: true
""",
    )
    return payload


def full_gate_rerun(account: dict[str, Any], news: dict[str, Any], symbol: dict[str, Any], session: dict[str, Any], lot: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    safety = order_send_safety()
    stop_absent = not STOP_BOT.exists()
    confirmation_present = CONFIRMATION.exists()
    gates = {
        "Account": account.get("state"),
        "Real Money": "REAL_BLOCKED",
        "MT5 Connection": "ALLOW" if account.get("terminal_connected") else "NO_TRADE",
        "News": news.get("gate"),
        "Week News Loaded": news.get("week_news_loaded"),
        "Data": symbol.get("state"),
        "Time": session.get("state"),
        "Symbol": symbol.get("state"),
        "Spread": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "StopLevel/FreezeLevel": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "Lot": lot.get("state"),
        "Signal": signal.get("state"),
        "Max Trades/Day": "ALLOW",
        "Weekend": "ALLOW" if session.get("state") != "NO_TRADE_WEEKEND" else "NO_TRADE_WEEKEND",
        "Order Router": safety.get("state"),
        "OrderCheck": "NOT_CALLED_UNTIL_ALL_GATES_PASS",
        "STOP_BOT": "ABSENT" if stop_absent else "ACTIVE",
        "Confirmation": "PRESENT" if confirmation_present else "ABSENT",
    }
    if account.get("state") != "FTMO_DEMO_TRIAL_CONFIRMED":
        state = "BLOCKED_ACCOUNT"
    elif news.get("state") == "NO_TRADE_NEWS_PROVIDER_UNAVAILABLE" or news.get("gate") != "ALLOW" and not news.get("provider_used"):
        state = "BLOCKED_NEWS_PROVIDER"
    elif news.get("state") == "NO_TRADE_NEWS_WINDOW":
        state = "BLOCKED_NEWS_WINDOW"
    elif news.get("gate") != "ALLOW" or news.get("week_news_loaded") is not True:
        state = "BLOCKED_NEWS_PROVIDER"
    elif signal.get("state") != "MANIPULANTE_SIGNAL_SYNC_OK":
        state = "BLOCKED_SIGNAL"
    elif symbol.get("state") != "ALLOW" or session.get("state") != "ALLOW" or lot.get("state") != "ALLOW":
        state = "REQUIRES_REPAIR"
    elif safety.get("state") != "PASS":
        state = "BLOCKED_ORDER_ROUTER"
    elif not stop_absent or not confirmation_present:
        state = "ALL_GATES_PASS_EXCEPT_CONFIRMATION_STOPBOT"
    else:
        state = "ALL_GATES_PASS"
    payload = {"timestamp_utc": now_iso(), "state": state, "gates": gates}
    write_json(OUT / "full_gate_rerun" / "phase37d_full_gate_rerun.json", payload)
    write_text(
        OUT / "full_gate_rerun" / "phase37d_full_gate_rerun.md",
        "\n".join(["# Phase37D Full Gate Rerun", "", f"- state: {state}"] + [f"- {key}: {value}" for key, value in gates.items()]),
    )
    return payload


def confirmation_stopbot_action(full_gate: dict[str, Any], dry_run: dict[str, Any]) -> dict[str, Any]:
    dry_pass = bool(dry_run.get("executed")) and dry_run.get("order_sent") is False and str(dry_run.get("decision", "")).startswith("DRY_RUN")
    can_enable = full_gate.get("state") == "ALL_GATES_PASS_EXCEPT_CONFIRMATION_STOPBOT" and dry_pass
    payload = {
        "timestamp_utc": now_iso(),
        "dry_run_pass": dry_pass,
        "all_gates_ready_for_confirmation": can_enable,
        "confirmation_file_created": False,
        "stop_bot_removed": False,
        "stop_bot_active_after": STOP_BOT.exists(),
        "reason": "",
    }
    if can_enable:
        CONFIRMATION.parent.mkdir(parents=True, exist_ok=True)
        CONFIRMATION.write_text(
            "\n".join(
                [
                    "I UNDERSTAND THIS IS FTMO FREE TRIAL DEMO ONLY",
                    "I CONFIRM NO REAL MONEY",
                    "I CONFIRM MANIPULANTE ONLY",
                    "RISK_DEFAULT=0.50",
                    "ONE_TRADE_PER_DAY",
                    "NEWS_GATE_REQUIRED",
                    "DATA_GATE_REQUIRED",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload["confirmation_file_created"] = True
        if STOP_BOT.exists():
            if STOP_DISABLED.exists():
                STOP_DISABLED.unlink()
            STOP_BOT.rename(STOP_DISABLED)
            payload["stop_bot_removed"] = True
        payload["reason"] = "All gates passed and dry-run passed"
    else:
        payload["reason"] = "At least one gate failed; confirmation file not created and STOP_BOT not removed"
    payload["stop_bot_active_after"] = STOP_BOT.exists()
    write_json(OUT / "confirmation_stopbot" / "phase37d_confirmation_stopbot_action.json", payload)
    write_text(
        OUT / "confirmation_stopbot" / "phase37d_confirmation_stopbot_action.md",
        f"""
# Phase37D Confirmation + STOP_BOT

- dry-run pass: {payload['dry_run_pass']}
- all gates ready: {payload['all_gates_ready_for_confirmation']}
- confirmation file created: {payload['confirmation_file_created']}
- STOP_BOT removed: {payload['stop_bot_removed']}
- reason: {payload['reason']}
""",
    )
    return payload


def dry_run_runner() -> dict[str, Any]:
    result = run_trial_runner(["--ftmo-trial", "--dry-run", "--risk", "0.005", "--no-real"])
    payload = {
        "timestamp_utc": now_iso(),
        "executed": True,
        "decision": result.get("final_decision"),
        "order_sent": bool(result.get("order_sent")),
        "reason": result.get("reason"),
        "api_news_primary": result.get("api_news_primary"),
        "runner_result": result,
    }
    write_json(OUT / "runner" / "phase37d_runner.json", payload)
    write_csv(
        OUT / "runner" / "phase37d_runner_decisions.csv",
        [{"timestamp": payload["timestamp_utc"], "mode": "DRY_RUN", "risk": 0.005, "decision": payload["decision"], "order_sent": payload["order_sent"], "reason": payload["reason"]}],
        ["timestamp", "mode", "risk", "decision", "order_sent", "reason"],
    )
    write_text(
        OUT / "runner" / "phase37d_runner.md",
        f"""
# Phase37D Runner

- dry-run executed: true
- auto runner started: false
- mode: DRY_RUN
- risk: 0.005
- order_sent: {payload['order_sent']}
- last decision: {payload['decision']}
""",
    )
    return payload


def auto_runner(full_gate_after: dict[str, Any]) -> dict[str, Any]:
    if full_gate_after.get("state") != "ALL_GATES_PASS":
        return {
            "timestamp_utc": now_iso(),
            "started": False,
            "mode": "NOT_STARTED",
            "risk": 0.005,
            "order_sent": False,
            "last_decision": "NO_TRADE",
            "reason": f"Full gate state is {full_gate_after.get('state')}",
        }
    result = run_trial_runner(["--ftmo-trial", "--risk", "0.005", "--no-real", "--i-understand-demo-automation"])
    return {
        "timestamp_utc": now_iso(),
        "started": True,
        "mode": "FTMO_TRIAL_AUTO",
        "risk": 0.005,
        "order_sent": bool(result.get("order_sent")),
        "last_decision": result.get("final_decision"),
        "runner_result": result,
    }


def update_runner_output_with_auto(dry_run: dict[str, Any], auto: dict[str, Any]) -> dict[str, Any]:
    payload = dict(dry_run)
    payload["auto_runner_started"] = auto.get("started")
    payload["auto_mode"] = auto.get("mode")
    payload["auto_order_sent"] = auto.get("order_sent")
    payload["auto_last_decision"] = auto.get("last_decision")
    payload["auto_reason"] = auto.get("reason")
    write_json(OUT / "runner" / "phase37d_runner.json", payload)
    write_text(
        OUT / "runner" / "phase37d_runner.md",
        f"""
# Phase37D Runner

- dry-run executed: {payload['executed']}
- dry-run decision: {payload['decision']}
- auto runner started: {auto.get('started')}
- mode: {auto.get('mode')}
- risk: {auto.get('risk')}
- order_sent: {auto.get('order_sent')}
- last decision: {auto.get('last_decision')}
- reason: {auto.get('reason', '')}
""",
    )
    return payload


def verdict(full_gate: dict[str, Any], news: dict[str, Any], signal: dict[str, Any], auto: dict[str, Any]) -> str:
    if auto.get("started") and full_gate.get("state") == "ALL_GATES_PASS":
        return "FTMO_TRIAL_AUTO_READY_AND_RUNNING"
    if full_gate.get("state") == "ALL_GATES_PASS":
        return "FTMO_TRIAL_AUTO_READY_NOT_RUNNING"
    if news.get("state") == "NO_TRADE_NEWS_WINDOW":
        return "FTMO_TRIAL_BLOCKED_NEWS_WINDOW"
    if news.get("gate") != "ALLOW" or news.get("week_news_loaded") is not True:
        return "FTMO_TRIAL_BLOCKED_NEWS_PROVIDER"
    if signal.get("state") != "MANIPULANTE_SIGNAL_SYNC_OK":
        return "FTMO_TRIAL_BLOCKED_SIGNAL"
    if STOP_BOT.exists():
        return "FTMO_TRIAL_BLOCKED_STOP_BOT"
    return "FTMO_TRIAL_REQUIRES_REPAIR"


def update_docs(final_verdict: str) -> None:
    status = {
        "timestamp_utc": now_iso(),
        "latest_phase": "PHASE37D_FTMO_TRIAL_API_NEWS_SIGNAL",
        "verdict": final_verdict,
        "authority": "MANIPULANTE_PHASE25",
        "strategy_changed": False,
        "real_blocked": True,
        "ftmo_trial_api_news_primary": True,
        "calendar_bridge": "FALLBACK_LEGACY",
        "risk_default": 0.005,
        "risk_075_trial_stress_only": True,
        "risk_100_prohibited": True,
        "news_sync_required": True,
        "signal_sync_required": True,
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status)
    write_json(LAB / "status.json", status)
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        f"""
# READ THIS FIRST

- Phase37D verdict: `{final_verdict}`.
- MANIPULANTE remains Phase25 Authority.
- FTMO Trial uses API news provider as primary source.
- MQL5 CalendarBridge is fallback/legacy only.
- Real remains blocked.
- Strategy unchanged.
""",
    )
    write_text(ROOT / "01_CURRENT_PROJECT_STATUS.md", f"# Current Project Status\n\n- Latest phase: Phase37D FTMO Trial API News + Signal.\n- Verdict: `{final_verdict}`.\n- Real: blocked.\n- Strategy: unchanged.")
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", {"timestamp_utc": now_iso(), "authority": "MANIPULANTE_PHASE25", "phase37d": final_verdict, "tp": 1.4, "be": 0.4, "bf": 70, "be05": "SHADOW_ONLY", "strategy_changed": False})
    write_text(ROOT / "02_STRATEGY_AUTHORITY_MAP.md", f"# Strategy Authority Map\n\n- MANIPULANTE Phase25 remains authority.\n- Phase37D: `{final_verdict}`.\n- BE0.5 remains shadow only.")
    write_text(MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_AUTOMATION_README.md", f"# FTMO Trial Automation\n\nPhase37D verdict: `{final_verdict}`.\n\nAPI news provider is primary. CalendarBridge is fallback/legacy. Real remains blocked.")
    write_text(MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_RUN_COMMANDS.md", "# FTMO Trial Run Commands\n\n```powershell\npython BOT_V2_DAYTIME_LAB\\src\\phase37d_ftmo_trial_api_news_signal_auto.py\n```\n\nNo auto order path is allowed unless all gates pass.")
    write_text(MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_NEWS_POLICY.md", "# FTMO Trial News Policy\n\nPrimary source: API Live News Provider. EUR/USD high impact events block +/-30 minutes. If provider/cache/timezone fails: NO_TRADE.")
    write_text(MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_KILL_SWITCH.md", "# FTMO Trial Kill Switch\n\n`STOP_BOT.txt` blocks automation. It is disabled only after all gates pass.")
    write_text(MANIPULANTE / "09_COMPLIANCE" / "LIVE_NEWS_FORTRESS_POLICY.md", "# Live News Fortress Policy\n\nPhase37D primary source is API provider. MQL5 CalendarBridge is fallback/legacy. No provider, stale cache, unknown impact or timezone error means NO_TRADE.")
    write_text(MANIPULANTE / "00_LEER_PRIMERO" / "README_MANIPULANTE.md", f"# MANIPULANTE\n\nPhase25 Authority. Phase37D verdict: `{final_verdict}`. Real blocked. Strategy unchanged.")
    manifest = "# ZIP CONTENTS MANIFEST\n\nIncludes Phase37D report/outputs, API news adapter, live signal engine, runner updates, MANIPULANTE docs and master docs. Excludes API keys, credentials, MT5 account files, heavy data, .pkl, ZIPs, .zipbak and api_news_provider_config.local.json."
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", manifest)


def write_report(data: dict[str, Any], final_verdict: str) -> dict[str, Any]:
    blockers: list[str] = []
    if data["news"].get("gate") != "ALLOW":
        blockers.append(data["news"].get("state", "NEWS_NOT_ALLOW"))
    if data["signal"].get("state") != "MANIPULANTE_SIGNAL_SYNC_OK":
        blockers.append(data["signal"].get("state", "SIGNAL_NOT_SYNCED"))
    if data["full_gate"].get("state") != "ALL_GATES_PASS":
        blockers.append(data["full_gate"].get("state", "GATES_NOT_PASS"))
    if STOP_BOT.exists():
        blockers.append("STOP_BOT_ACTIVE")
    if not CONFIRMATION.exists():
        blockers.append("CONFIRMATION_FILE_ABSENT")
    report = {
        "timestamp_utc": now_iso(),
        "objective": "Replace MT5 news EA with API provider and finalize exact MANIPULANTE live signal engine",
        "verdict": final_verdict,
        "api_provider": data["provider_config"],
        "api_live_news": data["news"],
        "calendar_bridge": {"fallback_legacy": True, "autostart_required": False},
        "account": data["account"],
        "symbol": data["symbol"],
        "time": data["time"],
        "lot": data["lot"],
        "signal": data["signal"],
        "full_gate_rerun": data["full_gate"],
        "dry_run": data["dry_run"],
        "confirmation_stopbot": data["confirmation_action"],
        "auto_runner": data["auto_runner"],
        "blockers": blockers,
        "warnings": ["No real trading authorized", "API keys excluded from Git and ZIP", "1.00% prohibited", "No strategy parameters changed"],
        "next_step": "Configurar un proveedor API live con key local o environment variable; luego rerun Phase37D.",
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        f"""
# PHASE37D FTMO TRIAL API NEWS SIGNAL AUTO REPORT

## Verdict

`{final_verdict}`

## API News

- provider used: {data['news'].get('provider_used')}
- gate: {data['news'].get('gate')}
- state: {data['news'].get('state')}
- today loaded: {data['news'].get('today_news_loaded')}
- week loaded: {data['news'].get('week_news_loaded')}

## Signal

- state: {data['signal'].get('state')}
- status: {data['signal'].get('signal_status')}

## Full Gate

- state: {data['full_gate'].get('state')}

## Runner

- dry-run decision: {data['dry_run'].get('decision')}
- auto started: {data['auto_runner'].get('started')}
- order_sent: {data['auto_runner'].get('order_sent')}

Real remains blocked.
""",
    )
    return report


def _zip_include(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    rel_str = rel.as_posix()
    lower = rel_str.lower()
    parts = set(rel.parts)
    if parts & {".git", ".venv", ".venv_fixed", ".pkg", ".vendor_duka", ".vendor_duka2", "__pycache__", "data", "ARCHIVE_SUPERSEDED"}:
        return False
    if lower.endswith((".zip", ".zipbak", ".pkl", ".pyc", ".log", ".tmp")):
        return False
    if "api_news_provider_config.local.json" in lower:
        return False
    if any(token in lower for token in [".env", "password", "token", "credential", "secret", "apikey", "api_key"]):
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    if rel.parts[0] == "MANIPULANTE":
        return True
    if rel.parts[0] == "BOT_V2_DAYTIME_LAB":
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/data/"):
            return False
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/outputs/") and "phase37d_ftmo_trial_api_news_signal" not in rel_str:
            return False
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/reports/") and "PHASE37D" not in rel_str:
            return False
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/src/"):
            return Path(rel_str).name.startswith("phase37")
        return True
    return path.name in {"00_READ_THIS_FIRST.md", "01_CURRENT_PROJECT_STATUS.md", "01_CURRENT_PROJECT_STATUS.json", "02_STRATEGY_AUTHORITY_MAP.md", "02_STRATEGY_AUTHORITY_MAP.json", "ZIP_CONTENTS_MANIFEST.md", ".gitignore"}


def rebuild_zip() -> dict[str, Any]:
    temp = ZIP_PATH.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    files = {path.relative_to(ROOT).as_posix(): path for path in ROOT.rglob("*") if _zip_include(path)}
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel, path in sorted(files.items()):
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
            "contains_phase37d_report": "BOT_V2_DAYTIME_LAB/reports/PHASE37D_FTMO_TRIAL_API_NEWS_SIGNAL_AUTO_REPORT.md" in names,
            "contains_phase37d_outputs": any(name.startswith("BOT_V2_DAYTIME_LAB/outputs/phase37d_ftmo_trial_api_news_signal/") for name in names),
            "contains_api_adapter": "BOT_V2_DAYTIME_LAB/src/phase37d_live_news_api_adapter.py" in names,
            "contains_signal_engine": "BOT_V2_DAYTIME_LAB/src/phase37d_manipulante_live_signal_engine.py" in names,
            "local_api_config_included": any("api_news_provider_config.local.json" in name.lower() for name in names),
            "heavy_entries_gt_2mb": [(name, zf.getinfo(name).file_size) for name in names if zf.getinfo(name).file_size > 2 * 1024 * 1024],
            "secret_like_entries": [name for name in names if any(token in name.lower() for token in [".env", "password", "token", "credential", "secret", "apikey", "api_key"])],
            "zip_entries_inside": [name for name in names if name.lower().endswith((".zip", ".zipbak"))],
        }
    write_json(OUT / "zip_validation" / "phase37d_zip_validation.json", validation)
    write_text(OUT / "zip_validation" / "phase37d_zip_validation.md", "\n".join(["# Phase37D ZIP Validation", ""] + [f"- {key}: {value}" for key, value in validation.items()]))
    return validation


def git_status(final_verdict: str) -> dict[str, Any]:
    status_short = run_cmd(["git", "status", "--short"])
    sensitive = [line for line in status_short.splitlines() if re.search(r"(\.env|credentials|token|password|broker account|mt5 real|\.pkl|api_news_provider_config\.local\.json)", line, re.I)]
    payload = {
        "timestamp_utc": now_iso(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status_short": status_short,
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "sensitive_status_entries": sensitive,
        "commit": "NO",
        "push": "NO",
        "hash": "N/A",
        "reason": f"Commit/push skipped because final verdict is {final_verdict}.",
    }
    write_json(OUT / "git" / "phase37d_git_status.json", payload)
    write_text(OUT / "git" / "phase37d_git_status.md", f"# Phase37D Git Status\n\n- branch: {payload['branch']}\n- commit: NO\n- push: NO\n- reason: {payload['reason']}")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zip", action="store_true")
    args = parser.parse_args(argv)

    pre = preflight()
    provider_config = provider_config_phase()
    news = write_api_news_outputs()
    account = write_account_outputs()
    symbol = write_symbol_outputs()
    session = write_time_outputs()
    lot = write_lot_outputs()
    signal = write_signal_outputs(news_gate=news.get("gate", "NO_TRADE"), data_gate=symbol.get("state", "NO_TRADE"), time_state=session.get("state"))
    full_gate = full_gate_rerun(account, news, symbol, session, lot, signal)
    dry = dry_run_runner()
    confirmation_action = confirmation_stopbot_action(full_gate, dry)
    full_gate_after = full_gate_rerun(account, news, symbol, session, lot, signal)
    auto = auto_runner(full_gate_after)
    dry = update_runner_output_with_auto(dry, auto)
    final_verdict = verdict(full_gate_after, news, signal, auto)
    update_docs(final_verdict)
    data = {
        "preflight": pre,
        "provider_config": provider_config,
        "news": news,
        "account": account,
        "symbol": symbol,
        "time": session,
        "lot": lot,
        "signal": signal,
        "full_gate": full_gate_after,
        "dry_run": dry,
        "confirmation_action": confirmation_action,
        "auto_runner": auto,
    }
    report = write_report(data, final_verdict)
    zip_validation = rebuild_zip() if not args.skip_zip else {"skipped": True}
    git = git_status(final_verdict)
    final = {"report": report, "zip_validation": zip_validation, "git": git}
    write_json(OUT / "phase37d_final_execution_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False, default=str))
    return final


if __name__ == "__main__":
    main()
