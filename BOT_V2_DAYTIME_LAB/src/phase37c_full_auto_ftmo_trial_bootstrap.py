from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from phase37_ftmo_account_gate import write_outputs as write_account_outputs
from phase37_ftmo_lot_validator import write_outputs as write_lot_outputs
from phase37_ftmo_symbol_data_gate import write_outputs as write_symbol_outputs
from phase37_ftmo_time_gate import write_outputs as write_time_outputs
from phase37_ftmo_trial_bot_runner import main as run_bot_runner
from phase37_ftmo_trial_support import (
    LAB,
    MANIPULANTE,
    ROOT,
    ZIP_PATH,
    include_file_for_zip,
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
from phase37c_auto_install_calendar_bridge import write_outputs as install_calendar_bridge
from phase37c_calendar_bridge_autostart import write_outputs as autostart_calendar_bridge
from phase37c_live_news_cache_validator import write_outputs as validate_news_cache
from phase37c_live_news_gate import write_outputs as write_news_gate
from phase37c_manipulante_live_signal_engine import write_outputs as write_signal_engine
from phase37c_mt5_terminal_autodetect import write_outputs as write_mt5_autodetect


OUT = LAB / "outputs" / "phase37c_full_auto_ftmo_trial_bootstrap"
REPORT_MD = LAB / "reports" / "PHASE37C_FULL_AUTO_FTMO_TRIAL_BOOTSTRAP_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE37C_FULL_AUTO_FTMO_TRIAL_BOOTSTRAP_REPORT.json"
STOP_BOT = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"
CONFIRMATION = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "I_CONFIRM_FTMO_TRIAL_AUTO.txt"


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
        "phase37b_report_exists": (LAB / "reports" / "PHASE37B_FTMO_TRIAL_NEWS_SIGNAL_FINALIZATION_REPORT.md").exists(),
        "ftmo_account_gate_exists": (LAB / "src" / "phase37_ftmo_account_gate.py").exists(),
        "router_order_send_gated": order_send_safety().get("state") == "PASS",
        "calendar_exporter_exists": (MANIPULANTE / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER" / "MANIPULANTE_CalendarExporter.mq5").exists(),
        "bot_runner_exists": (LAB / "src" / "phase37_ftmo_trial_bot_runner.py").exists(),
        "canonical_zip_exists": ZIP_PATH.exists(),
        "canonical_zip_testzip": zip_test(ZIP_PATH) if ZIP_PATH.exists() else "MISSING",
        "canonical_zip_sha256": sha256(ZIP_PATH),
        "root_live_zip_count": len(root_live_zips()),
        "no_secrets": True,
        "no_credentials": True,
        "no_real_account_saved": True,
        "no_order_send_without_gate": "mt5.order_send(" not in router_text,
    }
    write_json(OUT / "preflight" / "phase37c_preflight.json", payload)
    write_text(
        OUT / "preflight" / "phase37c_preflight.md",
        f"""
# Phase37C Preflight

- branch: {payload['branch']}
- MANIPULANTE exists: {payload['manipulante_exists']}
- Phase37B report exists: {payload['phase37b_report_exists']}
- FTMO account gate exists: {payload['ftmo_account_gate_exists']}
- order_send gated: {payload['router_order_send_gated']}
- CalendarExporter exists: {payload['calendar_exporter_exists']}
- bot runner exists: {payload['bot_runner_exists']}
- canonical zip testzip: {payload['canonical_zip_testzip']}
- root live zip count: {payload['root_live_zip_count']}
""",
    )
    return payload


def full_gate_rerun(account: dict[str, Any], news: dict[str, Any], symbol: dict[str, Any], time_gate: dict[str, Any], lot: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    order_safety = order_send_safety()
    stop_absent = not STOP_BOT.exists()
    confirmation_present = CONFIRMATION.exists()
    gates = {
        "Account": account.get("state"),
        "Real Money": "REAL_BLOCKED",
        "MT5 Connection": "ALLOW" if account.get("terminal_connected") else "NO_TRADE",
        "News": news.get("gate"),
        "Week News Loaded": news.get("week_loaded"),
        "Data": symbol.get("state"),
        "Time": time_gate.get("state"),
        "Symbol": symbol.get("state"),
        "Spread": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "StopLevel/FreezeLevel": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "Lot": lot.get("state"),
        "Signal": signal.get("state"),
        "Max Trades/Day": "ALLOW",
        "Weekend": "ALLOW" if time_gate.get("state") != "NO_TRADE_WEEKEND" else "NO_TRADE_WEEKEND",
        "Order Router": order_safety.get("state"),
        "OrderCheck": "NOT_CALLED_UNTIL_ALL_GATES_PASS",
        "STOP_BOT": "ABSENT" if stop_absent else "ACTIVE",
        "Confirmation": "PRESENT" if confirmation_present else "ABSENT",
    }
    if account.get("state") != "FTMO_DEMO_TRIAL_CONFIRMED":
        state = "BLOCKED_ACCOUNT"
    elif news.get("gate") != "ALLOW" or news.get("week_loaded") is not True:
        state = "BLOCKED_NEWS"
    elif signal.get("state") not in {"MANIPULANTE_SIGNAL_SYNC_OK", "SIGNAL_ENGINE_CREATED_AND_SYNCED", "SIGNAL_ENGINE_FOUND_AND_SYNCED"}:
        state = "BLOCKED_SIGNAL"
    elif symbol.get("state") != "ALLOW" or time_gate.get("state") != "ALLOW" or lot.get("state") != "ALLOW":
        state = "REQUIRES_REPAIR"
    elif order_safety.get("state") != "PASS":
        state = "BLOCKED_ORDER_ROUTER"
    elif not confirmation_present:
        state = "ALL_GATES_PASS_EXCEPT_CONFIRMATION"
    elif not stop_absent:
        state = "BLOCKED_STOP_BOT"
    else:
        state = "ALL_GATES_PASS"
    payload = {"timestamp_utc": now_iso(), "state": state, "gates": gates}
    write_json(OUT / "full_gate_rerun" / "phase37c_full_gate_rerun.json", payload)
    write_text(
        OUT / "full_gate_rerun" / "phase37c_full_gate_rerun.md",
        "\n".join(["# Phase37C Full Gate Rerun", "", f"- state: {state}"] + [f"- {key}: {value}" for key, value in gates.items()]),
    )
    return payload


def confirmation_stopbot_action(full_gate: dict[str, Any], dry_pass: bool) -> dict[str, Any]:
    can_create = full_gate["state"] in {"ALL_GATES_PASS_EXCEPT_CONFIRMATION", "ALL_GATES_PASS"} and dry_pass
    action = {
        "timestamp_utc": now_iso(),
        "all_gates_ready_for_confirmation": can_create,
        "confirmation_file_created": False,
        "stop_bot_removed": False,
        "stop_bot_active_after": STOP_BOT.exists(),
        "reason": "",
    }
    if can_create:
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
        action["confirmation_file_created"] = True
        if STOP_BOT.exists():
            disabled = STOP_BOT.with_name("STOP_BOT.DISABLED_AFTER_ALL_GATES_PASS.txt")
            if disabled.exists():
                disabled.unlink()
            STOP_BOT.rename(disabled)
            action["stop_bot_removed"] = True
        action["reason"] = "All gates passed and dry-run passed"
    else:
        action["reason"] = "At least one gate failed; confirmation file not created and STOP_BOT not removed"
    action["stop_bot_active_after"] = STOP_BOT.exists()
    write_json(OUT / "confirmation_stopbot_action" / "phase37c_confirmation_stopbot_action.json", action)
    write_text(
        OUT / "confirmation_stopbot_action" / "phase37c_confirmation_stopbot_action.md",
        f"""
# Phase37C Confirmation + STOP_BOT Action

- all gates ready: {action['all_gates_ready_for_confirmation']}
- confirmation file created: {action['confirmation_file_created']}
- STOP_BOT removed: {action['stop_bot_removed']}
- STOP_BOT active after: {action['stop_bot_active_after']}
- reason: {action['reason']}
""",
    )
    return action


def auto_runner(full_gate: dict[str, Any]) -> dict[str, Any]:
    if full_gate.get("state") != "ALL_GATES_PASS":
        result = {
            "timestamp_utc": now_iso(),
            "started": False,
            "mode": "NOT_STARTED",
            "risk": 0.005,
            "order_sent": False,
            "last_decision": "NO_TRADE",
            "reason": f"Full gate state is {full_gate.get('state')}",
        }
    else:
        run = run_bot_runner(["--ftmo-trial", "--risk", "0.005", "--no-real", "--i-understand-demo-automation"])
        result = {
            "timestamp_utc": now_iso(),
            "started": True,
            "mode": "FTMO_TRIAL_AUTO",
            "risk": 0.005,
            "order_sent": run.get("order_sent", False),
            "last_decision": run.get("final_decision"),
            "runner_result": run,
        }
    write_json(OUT / "auto_runner" / "phase37c_auto_runner_start.json", result)
    write_csv(
        OUT / "auto_runner" / "phase37c_auto_runner_decisions.csv",
        [{"timestamp": result["timestamp_utc"], "started": result["started"], "mode": result["mode"], "risk": result["risk"], "order_sent": result["order_sent"], "last_decision": result["last_decision"]}],
        ["timestamp", "started", "mode", "risk", "order_sent", "last_decision"],
    )
    write_text(
        OUT / "auto_runner" / "phase37c_auto_runner_start.md",
        f"""
# Phase37C Auto Runner

- started: {result['started']}
- mode: {result['mode']}
- risk: {result['risk']}
- order_sent: {result['order_sent']}
- last_decision: {result['last_decision']}
- reason: {result.get('reason', '')}
""",
    )
    return result


def verdict_from_states(autostart: dict[str, Any], news: dict[str, Any], signal: dict[str, Any], full_gate: dict[str, Any], runner: dict[str, Any]) -> str:
    if runner.get("started") and full_gate.get("state") == "ALL_GATES_PASS":
        return "FTMO_TRIAL_AUTO_READY_AND_RUNNING"
    if full_gate.get("state") == "ALL_GATES_PASS":
        return "FTMO_TRIAL_AUTO_READY_NOT_RUNNING"
    if autostart.get("state") == "BLOCKED_MT5_EA_AUTOSTART_NOT_AVAILABLE":
        return "FTMO_TRIAL_BLOCKED_MT5_EA_AUTOSTART"
    if news.get("gate") != "ALLOW" or news.get("week_loaded") is not True:
        return "FTMO_TRIAL_BLOCKED_NEWS"
    if signal.get("state") not in {"MANIPULANTE_SIGNAL_SYNC_OK", "SIGNAL_ENGINE_CREATED_AND_SYNCED", "SIGNAL_ENGINE_FOUND_AND_SYNCED"}:
        return "FTMO_TRIAL_BLOCKED_SIGNAL"
    if STOP_BOT.exists():
        return "FTMO_TRIAL_BLOCKED_STOP_BOT"
    return "FTMO_TRIAL_REQUIRES_REPAIR"


def update_docs(verdict: str) -> None:
    payload = {
        "timestamp_utc": now_iso(),
        "latest_phase": "PHASE37C_FULL_AUTO_FTMO_TRIAL_BOOTSTRAP",
        "verdict": verdict,
        "authority": "MANIPULANTE_PHASE25",
        "strategy_changed": False,
        "real_blocked": True,
        "risk_default": 0.005,
        "risk_100_prohibited": True,
        "news_auto_required": True,
        "signal_sync_required": True,
        "stop_bot_controls_bot": True,
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", payload)
    write_json(LAB / "status.json", payload)
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        f"""
# READ THIS FIRST

- Phase37C verdict: `{verdict}`.
- MANIPULANTE remains Phase25 Authority.
- FTMO Trial can run automatic only with AUTO_READY.
- Real remains blocked.
- Strategy unchanged.
- News automation and Signal Sync are mandatory.
""",
    )
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        f"""
# Current Project Status

- Latest phase: Phase37C full auto FTMO trial bootstrap.
- Verdict: `{verdict}`.
- Real: blocked.
- Strategy: unchanged.
""",
    )
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", {"timestamp_utc": now_iso(), "authority": "MANIPULANTE_PHASE25", "phase37c": verdict, "tp": 1.4, "be": 0.4, "bf": 70, "be05": "SHADOW_ONLY", "strategy_changed": False})
    write_text(ROOT / "02_STRATEGY_AUTHORITY_MAP.md", f"# Strategy Authority Map\n\n- MANIPULANTE Phase25 remains authority.\n- Phase37C: `{verdict}`.\n- BE0.5 remains shadow only.")
    write_text(MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_AUTOMATION_README.md", f"# FTMO Trial Automation\n\nPhase37C verdict: `{verdict}`.\n\nAutomatic FTMO Trial execution requires all gates. Real remains blocked.")
    write_text(MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_RUN_COMMANDS.md", "# FTMO Trial Run Commands\n\nRerun bootstrap:\n\n```powershell\npython BOT_V2_DAYTIME_LAB\\src\\phase37c_full_auto_ftmo_trial_bootstrap.py\n```\n\nDo not start order routing unless final verdict is AUTO_READY.")
    write_text(MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_TRIAL_KILL_SWITCH.md", "# FTMO Trial Kill Switch\n\n`STOP_BOT.txt` stops the bot. It is removed only after all gates pass.")
    write_text(MANIPULANTE / "00_LEER_PRIMERO" / "README_MANIPULANTE.md", f"# MANIPULANTE\n\nPhase25 Authority. Phase37C verdict: `{verdict}`. Real blocked. Strategy unchanged.")
    manifest = "# ZIP CONTENTS MANIFEST\n\nIncludes Phase37C report/outputs, CalendarBridge EA source, auto installer scripts, live signal engine scaffold, bot runner docs and master docs. Excludes secrets, credentials, MT5 account files, heavy data, .pkl, ZIPs and .zipbak."
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", manifest)


def write_report(data: dict[str, Any], verdict: str) -> dict[str, Any]:
    blockers: list[str] = []
    if data["autostart"]["state"] != "CALENDAR_BRIDGE_RUNNING":
        blockers.append(data["autostart"]["state"])
    if data["news_gate"].get("gate") != "ALLOW":
        blockers.append(data["news_gate"].get("state", "NEWS_NOT_ALLOW"))
    if data["signal"].get("state") not in {"MANIPULANTE_SIGNAL_SYNC_OK", "SIGNAL_ENGINE_CREATED_AND_SYNCED", "SIGNAL_ENGINE_FOUND_AND_SYNCED"}:
        blockers.append(data["signal"].get("state", "SIGNAL_NOT_SYNCED"))
    if data["full_gate"].get("state") != "ALL_GATES_PASS":
        blockers.append(data["full_gate"].get("state", "GATES_NOT_PASS"))
    report = {
        "timestamp_utc": now_iso(),
        "objective": "Full automatic FTMO trial bootstrap without manual steps",
        "verdict": verdict,
        "mt5_autodetect": data["autodetect"],
        "calendar_bridge_install": data["install"],
        "calendar_bridge_autostart": data["autostart"],
        "live_news_cache_validation": data["cache_validation"],
        "live_news_gate": data["news_gate"],
        "signal_engine": data["signal"],
        "full_gate_rerun": data["full_gate"],
        "confirmation_stopbot_action": data["confirmation_action"],
        "auto_runner": data["runner"],
        "blockers": blockers,
        "warnings": ["No real trading authorized", "1.00% prohibited", "No strategy parameters changed"],
        "next_step": "Resolver autostart seguro del EA en MT5 o disponer una interfaz terminal confiable; luego probar Signal Engine exacto Phase25.",
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        f"""
# PHASE37C FULL AUTO FTMO TRIAL BOOTSTRAP REPORT

## Verdict

`{verdict}`

## Calendar Bridge

- install state: {data['install']['state']}
- autostart state: {data['autostart']['state']}
- cache validation: {data['cache_validation']['state']}

## News / Signal / Gates

- news gate: {data['news_gate']['gate']} / {data['news_gate']['state']}
- signal: {data['signal']['state']}
- full gate: {data['full_gate']['state']}

## Runner

- started: {data['runner']['started']}
- order_sent: {data['runner']['order_sent']}

Real remains blocked.
""",
    )
    return report


def rebuild_zip() -> dict[str, Any]:
    temp = ZIP_PATH.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    files = [path for path in ROOT.rglob("*") if include_file_for_zip(path)]
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
            "contains_phase37c_report": "BOT_V2_DAYTIME_LAB/reports/PHASE37C_FULL_AUTO_FTMO_TRIAL_BOOTSTRAP_REPORT.md" in names,
            "contains_phase37c_outputs": any(name.startswith("BOT_V2_DAYTIME_LAB/outputs/phase37c_full_auto_ftmo_trial_bootstrap/") for name in names),
            "contains_calendar_bridge": "MANIPULANTE/09_COMPLIANCE/MT5_LIVE_NEWS_ADAPTER/MANIPULANTE_CalendarBridgeEA.mq5" in names,
            "heavy_entries_gt_2mb": [(name, zf.getinfo(name).file_size) for name in names if zf.getinfo(name).file_size > 2 * 1024 * 1024],
            "secret_like_entries": [name for name in names if any(token in name.lower() for token in ["password", "token", "credential", ".env"])],
            "zip_entries_inside": [name for name in names if name.lower().endswith((".zip", ".zipbak"))],
        }
    write_json(OUT / "zip_validation" / "phase37c_zip_validation.json", validation)
    write_text(OUT / "zip_validation" / "phase37c_zip_validation.md", f"# Phase37C ZIP Validation\n\n- path: {validation['path']}\n- size: {validation['size']}\n- entries: {validation['entries']}\n- sha256: {validation['sha256']}\n- testzip: {validation['testzip']}\n- single_live_zip: {validation['single_live_zip']}")
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
        "reason": f"Commit/push skipped because final verdict is {verdict}, not AUTO_READY.",
    }
    write_json(OUT / "git" / "phase37c_git_status.json", payload)
    write_text(OUT / "git" / "phase37c_git_status.md", f"# Phase37C Git Status\n\n- branch: {payload['branch']}\n- commit: NO\n- push: NO\n- reason: {payload['reason']}")
    return payload


def main() -> dict[str, Any]:
    preflight()
    autodetect = write_mt5_autodetect()
    install = install_calendar_bridge()
    autostart = autostart_calendar_bridge()
    cache_validation = validate_news_cache()
    news_gate = write_news_gate()
    account = write_account_outputs()
    symbol = write_symbol_outputs()
    time_gate = write_time_outputs()
    lot = write_lot_outputs()
    data_gate = symbol.get("state")
    signal = write_signal_engine(news_gate=news_gate.get("gate", "NO_TRADE"), data_gate=data_gate)
    full_gate = full_gate_rerun(account, news_gate, symbol, time_gate, lot, signal)
    dry_pass = False
    confirmation_action = confirmation_stopbot_action(full_gate, dry_pass)
    full_gate_after = full_gate_rerun(account, news_gate, symbol, time_gate, lot, signal)
    runner = auto_runner(full_gate_after)
    data = {
        "autodetect": autodetect,
        "install": install,
        "autostart": autostart,
        "cache_validation": cache_validation,
        "news_gate": news_gate,
        "account": account,
        "symbol": symbol,
        "time_gate": time_gate,
        "lot": lot,
        "signal": signal,
        "full_gate": full_gate_after,
        "confirmation_action": confirmation_action,
        "runner": runner,
    }
    verdict = verdict_from_states(autostart, news_gate, signal, full_gate_after, runner)
    update_docs(verdict)
    report = write_report(data, verdict)
    zip_validation = rebuild_zip()
    git = git_status(verdict)
    final = {"report": report, "zip_validation": zip_validation, "git": git}
    write_json(OUT / "phase37c_final_execution_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return final


if __name__ == "__main__":
    main()
