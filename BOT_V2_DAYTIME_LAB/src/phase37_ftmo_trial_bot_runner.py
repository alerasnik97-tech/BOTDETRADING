from __future__ import annotations

import argparse
import csv
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_order_router import evaluate_gates
from phase37_ftmo_trial_support import (
    MANIPULANTE,
    NY,
    AR,
    OUT,
    account_gate,
    confirmation_file_status,
    detect_symbol,
    lot_gate_10k,
    order_send_safety,
    signal_sync,
    time_gate,
    write_json,
    write_text,
)

# New Phase37X Modules
import phase37x_session_lifecycle as lifecycle
import phase37x_position_state as pos_state
import phase37x_safe_close as safe_close


STOP_FILE = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"
LOG_DIR = MANIPULANTE / "10_LOGS_PAPER" / "ftmo_trial_bot"
DECISION_LOG = LOG_DIR / "decisions.csv"
HEARTBEAT_JSON = LOG_DIR / "heartbeat.json"
HEARTBEAT_TXT = LOG_DIR / "heartbeat.txt"
LOCK_FILE = LOG_DIR / "runner.lock"


def append_decision(row: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    exists = DECISION_LOG.exists()
    fields = [
        "timestamp_local",
        "timestamp_ny",
        "final_decision",
        "reason",
        "signal_status",
        "order_sent",
        "account",
        "news_gate",
        "data_gate",
        "time_gate",
        "session_state",
        "position_state",
        "gates_status",
    ]
    with DECISION_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fields})


def write_heartbeat(result: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pos = result.get("position_state_info", {})
    cfg = lifecycle.load_config()
    hb = {
        "timestamp_local": datetime.now(AR).isoformat(),
        "timestamp_ny": datetime.now(NY).isoformat(),
        "pid": os.getpid(),
        "account_company": result.get("account_company"),
        "server": result.get("server"),
        "account_mode": result.get("account_mode"),
        "symbol": result.get("symbol"),
        "session_state": result.get("session_state"),
        "can_open_new_trades": result.get("can_open_new_trades"),
        "position_state": pos.get("state"),
        "position_ticket": pos.get("ticket"),
        "position_side": pos.get("side"),
        "position_has_sl": pos.get("has_sl"),
        "position_has_tp": pos.get("has_tp"),
        "manage_only": result.get("session_state") in {"MANAGE_ONLY", "USER_PC_OFF_PROTECTION_WINDOW"},
        "forced_safe_close_time_ny": cfg.get("forced_safe_close"),
        "verify_flat_time_ny": cfg.get("verify_flat"),
        "shutdown_time_ny": cfg.get("daily_shutdown"),
        "news_gate": result.get("gates", {}).get("api_live_news_gate"),
        "data_gate": result.get("gates", {}).get("data_quality_gate"),
        "time_gate": result.get("gates", {}).get("time_gate"),
        "signal_status": result.get("signal_sync", {}).get("state"),
        "last_decision": result.get("final_decision"),
        "next_news_block": result.get("next_news_block"),
        "order_sent": result.get("order_sent"),
        "runner_status": "RUNNING",
        "pc_off_warning": result.get("session_state") == "USER_PC_OFF_PROTECTION_WINDOW",
        "critical_do_not_turn_off_pc": result.get("critical_do_not_turn_off_pc", False),
        "shutdown_allowed": result.get("shutdown_allowed", False),
        "last_safe_close_status": result.get("safe_close_result", {}).get("status") if result.get("safe_close_result") else None,
    }
    write_json(HEARTBEAT_JSON, hb)
    lines = [f"{k}: {v}" for k, v in hb.items()]
    write_text(HEARTBEAT_TXT, "\n".join(lines))


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    api_mode = False
    try:
        from phase37d_live_news_api_adapter import news_gate_status
        from phase37d_manipulante_live_signal_engine import evaluate_live_signal

        api_mode = True
        account = account_gate()
        symbol = detect_symbol()
        session_gate = time_gate(symbol)
        news = news_gate_status(force_fetch=True)
        lot = lot_gate_10k(symbol, account)
        safety = order_send_safety()
        confirmation = confirmation_file_status()
        
        # Phase37X Modules Integration
        pos_info = pos_state.get_position_state(symbol.get("symbol", "EURUSD"))
        session_state = lifecycle.get_session_state(position_open=(pos_info["state"] != "FLAT"))
        can_open = lifecycle.can_open_new_trades()
        
        signal = evaluate_live_signal(
            news_gate=news.get("gate", "NO_TRADE"),
            data_gate=symbol.get("state", "NO_TRADE"),
            time_state=session_gate.get("state"),
        )
        
        gates = {
            "account_gate": account.get("state"),
            "real_money_gate": "REAL_BLOCKED",
            "mt5_connection_gate": "ALLOW" if account.get("terminal_connected") else "NO_TRADE",
            "api_live_news_gate": news.get("gate"),
            "week_news_loaded": news.get("week_news_loaded"),
            "data_quality_gate": symbol.get("state"),
            "time_gate": session_gate.get("state"),
            "symbol_gate": symbol.get("state"),
            "spread_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
            "stoplevel_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
            "lot_gate": lot.get("state"),
            "signal_engine_gate": signal.get("state"),
            "max_trades_day_gate": "ALLOW",
            "weekend_gate": "ALLOW" if session_gate.get("state") != "NO_TRADE_WEEKEND" else "NO_TRADE_WEEKEND",
            "order_router_safety": safety.get("state"),
            "confirmation_file": confirmation.get("valid"),
            "lifecycle_state": session_state,
        }
        
        final_decision = "DRY_RUN_NO_SIGNAL" if args.dry_run else "NO_TRADE"
        reason = signal.get("signal_status", "NO_SIGNAL")
        
        # Lifecycle Logic Overrides
        if session_state == "BEFORE_SESSION_WAIT":
            final_decision = "WAITING_FOR_SESSION"
            reason = "Market session not yet open (pre-07:00 NY)"
        elif not can_open and pos_info["state"] == "FLAT":
            final_decision = "NO_NEW_TRADES_AFTER_CUTOFF"
            reason = "Trading window closed for new entries"
        elif session_state == "WEEKEND_BLOCK":
            final_decision = "NO_TRADE_WEEKEND"
            reason = "Weekend market closed"
        
        # Normal Checks if session active
        if final_decision not in {"WAITING_FOR_SESSION", "NO_NEW_TRADES_AFTER_CUTOFF", "NO_TRADE_WEEKEND"}:
            checks = [
                (account.get("state") == "FTMO_DEMO_TRIAL_CONFIRMED", "NO_TRADE_ACCOUNT"),
                (news.get("gate") == "ALLOW", "NO_TRADE_NEWS_BLOCK"),
                (news.get("week_news_loaded") is True, "NO_TRADE_WEEK_NEWS_NOT_LOADED"),
                (symbol.get("state") == "ALLOW", "NO_TRADE_DATA_OR_SYMBOL"),
                (session_gate.get("state") == "ALLOW", "NO_TRADE_TIME"),
                (lot.get("state") == "ALLOW", "NO_TRADE_LOT"),
                (signal.get("state") == "MANIPULANTE_SIGNAL_SYNC_OK", "NO_TRADE_SIGNAL_SYNC"),
                (safety.get("state") == "PASS", "NO_TRADE_ORDER_ROUTER"),
                (confirmation.get("valid") is True, "CONFIRMATION_MISSING"),
            ]
            for ok, fail_reason in checks:
                if not ok:
                    final_decision = fail_reason
                    reason = fail_reason
                    break
            
            if final_decision == "DRY_RUN_NO_SIGNAL" and signal.get("signal_status") == "SIGNAL_READY":
                final_decision = "DRY_RUN_ALLOW_SIGNAL_READY"
                reason = "DRY_RUN_NO_ORDER_SEND"
            elif final_decision == "NO_TRADE" and signal.get("signal_status") == "SIGNAL_READY":
                # Only allow if we are in SESSION_ACTIVE
                if session_state == "SESSION_ACTIVE":
                    final_decision = "ALLOW"
                    reason = "SIGNAL_READY_GATES_ALLOW"
                else:
                    final_decision = "NO_TRADE_OUTSIDE_ENTRY_WINDOW"
                    reason = f"Signal ready but session state is {session_state}"

        # Forced Safe Close Execution
        safe_close_res = None
        if not args.dry_run and lifecycle.should_force_safe_close() and pos_info["state"] != "FLAT":
            print(f"[CRITICAL] Forced safe close triggered by lifecycle: {session_state}")
            safe_close_res = safe_close.execute_safe_close(symbol.get("symbol", "EURUSD"))
            final_decision = "FORCED_CLOSE_EXECUTED"
            reason = safe_close_res.get("status")
            # Refresh position info after close
            pos_info = pos_state.get_position_state(symbol.get("symbol", "EURUSD"))

        shutdown_allowed = False
        critical_pc = False
        if lifecycle.should_daily_shutdown():
            if pos_info["state"] == "FLAT":
                shutdown_allowed = True
            else:
                critical_pc = True
                final_decision = "CRITICAL_DO_NOT_TURN_OFF_PC"
                reason = "Position still open during shutdown window"

        result = {
            "timestamp_ny": datetime.now(NY).isoformat(),
            "timestamp_local": datetime.now(AR).isoformat(),
            "final_decision": final_decision,
            "reason": reason,
            "order_sent": False,
            "account_company": account.get("company"),
            "server": account.get("server"),
            "account_mode": account.get("trade_mode_label"),
            "symbol": symbol.get("symbol"),
            "session_state": session_state,
            "can_open_new_trades": can_open,
            "position_state_info": pos_info,
            "next_news_block": news.get("next_blocking_event", {}).get("event_time_ny") if news.get("next_blocking_event") else None,
            "signal_sync": signal,
            "gates": gates,
            "api_news_primary": api_mode,
            "shutdown_allowed": shutdown_allowed,
            "critical_do_not_turn_off_pc": critical_pc,
            "safe_close_result": safe_close_res,
        }
        
        append_decision({
            **result,
            "account": f"{result['account_company']} ({result['account_mode']})",
            "news_gate": gates["api_live_news_gate"],
            "data_gate": gates["data_quality_gate"],
            "time_gate": gates["time_gate"],
            "session_state": session_state,
            "position_state": pos_info["state"],
            "gates_status": json.dumps(gates, ensure_ascii=False),
            "signal_status": signal.get("state"),
        })
        write_heartbeat(result)
        return result

    except Exception as exc:
        print(f"Error in run_once: {exc}")
        import traceback
        traceback.print_exc()
        return {"final_decision": "ERROR", "reason": str(exc), "order_sent": False}


def write_dry_run_outputs(result: dict[str, Any]) -> None:
    out_dir = OUT / "dry_run"
    write_json(out_dir / "phase37_dry_run.json", result)
    write_text(
        out_dir / "phase37_dry_run.md",
        f"""
# Phase37 FTMO Trial Dry-Run
- decision: {result.get('final_decision')}
- reason: {result.get('reason')}
- session: {result.get('session_state')}
- order_sent: {result.get('order_sent')}
""",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MANIPULANTE FTMO trial bot runner")
    parser.add_argument("--ftmo-trial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--i-understand-demo-automation", action="store_true", default=True)
    parser.add_argument("--no-real", action="store_true", default=True)
    parser.add_argument("--once", action="store_true", default=False)
    parser.add_argument("--interval-seconds", type=int, default=60)
    return parser


def safe_dumps(obj: Any) -> str:
    def default(o: Any) -> str:
        if hasattr(o, "isoformat"): return o.isoformat()
        if hasattr(o, "item"): return o.item()
        return str(o)
    return json.dumps(obj, default=default, indent=2, ensure_ascii=False)


def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            if os.name == 'nt':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(1, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return False
        except Exception:
            pass
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    if not args.dry_run and not acquire_lock():
        print("DUPLICATE_RUNNER_DETECTION: Another instance is running. Aborting.")
        sys.exit(1)

    try:
        if args.once or args.dry_run:
            result = run_once(args)
            if args.dry_run: write_dry_run_outputs(result)
            print(safe_dumps(result))
            return result
        
        print(f"MANIPULANTE FTMO TRIAL AUTO-RUNNER STARTED [PID {os.getpid()}]")
        last: dict[str, Any] = {}
        while not STOP_FILE.exists():
            last = run_once(args)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Decision: {last.get('final_decision')} | Reason: {last.get('reason')} | Session: {last.get('session_state')}")
            
            if last.get("shutdown_allowed"):
                print("DAILY_AUTO_SHUTDOWN reached and FLAT. Shutting down runner.")
                break
            
            time.sleep(max(10, args.interval_seconds))
        
        if not last:
            last = {"final_decision": "STOP_BOT_ACTIVE", "order_sent": False}
        print("Runner loop ended safely.")
        return last
    finally:
        if not args.dry_run: release_lock()


if __name__ == "__main__":
    main()
