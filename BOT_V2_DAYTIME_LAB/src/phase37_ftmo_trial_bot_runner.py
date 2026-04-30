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
    order_send_readiness_audit,
    order_send_safety,
    signal_sync,
    time_gate,
    write_json,
    write_text,
)

# Phase37X Modules
import phase37x_session_lifecycle as lifecycle
import phase37x_position_state as pos_state
import phase37x_safe_close as safe_close


STOP_FILE = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"
LOG_DIR = MANIPULANTE / "10_LOGS_PAPER" / "ftmo_trial_bot"
DECISION_LOG = LOG_DIR / "decisions.csv"
HEARTBEAT_JSON = LOG_DIR / "heartbeat.json"
HEARTBEAT_TXT = LOG_DIR / "heartbeat.txt"
QUICK_STATUS_TXT = LOG_DIR / "quick_status.txt"
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
        "manage_only": lifecycle.should_manage_only(),
        "pc_off_deadline_ny": cfg.get("pc_must_be_off_after"),
        "forced_safe_close_time_ny": cfg.get("forced_safe_close"),
        "news_gate": result.get("gates", {}).get("api_live_news_gate"),
        "data_gate": result.get("gates", {}).get("data_quality_gate"),
        "time_gate": result.get("gates", {}).get("time_gate"),
        "signal_status": result.get("signal_sync", {}).get("state"),
        "last_decision": result.get("final_decision"),
        "next_news_block": result.get("next_news_block"),
        "order_sent": result.get("order_sent"),
        "runner_status": "RUNNING",
        "forced_close_attempts": result.get("safe_close_result", {}).get("attempts") if result.get("safe_close_result") else 0,
        "last_close_attempt_status": result.get("safe_close_result", {}).get("status") if result.get("safe_close_result") else None,
        "flat_confirmed_1950": result.get("flat_confirmed_1950", False),
        "flat_confirmed_1955": result.get("flat_confirmed_1955", False),
        "safe_to_turn_off_pc": result.get("shutdown_allowed", False),
        "manual_intervention_required": result.get("manual_intervention_required", False),
        "critical_position_still_open": result.get("critical_do_not_turn_off_pc", False),
        "order_readiness_state": result.get("order_readiness", {}).get("state"),
        "order_check_retcode": result.get("order_readiness", {}).get("order_check_retcode"),
        "order_check_pass": result.get("order_readiness", {}).get("order_check_pass"),
        "account_trade_allowed": result.get("order_readiness", {}).get("account_trade_allowed"),
        "account_trade_expert": result.get("order_readiness", {}).get("account_trade_expert"),
        "terminal_connected": result.get("order_readiness", {}).get("terminal_connected"),
        "terminal_trade_allowed": result.get("order_readiness", {}).get("terminal_trade_allowed"),
        "tradeapi_disabled": result.get("order_readiness", {}).get("tradeapi_disabled"),
        "orders_message": result.get("order_readiness", {}).get("orders_message"),
        "action_required": result.get("order_readiness", {}).get("action_required"),
        "permission_conclusion": result.get("order_readiness", {}).get("permission_conclusion"),
    }
    write_json(HEARTBEAT_JSON, hb)
    lines = [f"{k}: {v}" for k, v in hb.items()]
    write_text(HEARTBEAT_TXT, "\n".join(lines))
    
    # Phase 37ZF quick status, ASCII only for Windows CMD.
    estado_gen = "OK - BOT ACTIVO"
    msg_gen = "BOT ACTIVO"
    operacion_abierta = "SI" if hb["position_state"] != "FLAT" else "NO"
    seguro_apagar = "SI" if hb["safe_to_turn_off_pc"] else "NO"
    
    if hb["news_gate"] != "ALLOW" or not hb["can_open_new_trades"]:
        estado_gen = "BLOQUEADO - BOT ACTIVO PERO NO OPERA"
        msg_gen = "BOT ACTIVO PERO NO OPERA POR REGLA"

    if hb["order_readiness_state"] == "BLOCKED_AUTOTRADING_DISABLED":
        estado_gen = "BLOQUEADO - BOT ACTIVO PERO NO OPERA"
        msg_gen = "AUTOTRADING DESHABILITADO"
        
    if operacion_abierta == "SI" or hb["critical_position_still_open"] or hb["manual_intervention_required"]:
        estado_gen = "PELIGRO - NO APAGAR PC"
        msg_gen = "NO APAGAR PC"
    
    qs_lines = [
        f"ESTADO_GENERAL={estado_gen}",
        f"MENSAJE={msg_gen}",
        f"CUENTA={hb['server'] or 'FTMO-Demo'}",
        f"RUNNER=ACTIVO",
        f"NEWS={hb['news_gate'] or 'NO_TRADE'}",
        f"MOTIVO={'AUTOTRADING_DESHABILITADO' if hb['order_readiness_state'] == 'BLOCKED_AUTOTRADING_DISABLED' else ''}",
        f"ORDENES={hb['orders_message'] or 'ORDENES: DESCONOCIDAS'}",
        f"ORDER_CHECK={'PASS' if hb['order_check_pass'] else 'FAIL'}",
        f"ORDER_CHECK_RETCODE={hb['order_check_retcode'] if hb['order_check_retcode'] is not None else '---'}",
        f"ORDER_SEND=GATEADO",
        f"ACCOUNT_TRADE_ALLOWED={'SI' if hb['account_trade_allowed'] else 'NO'}",
        f"ACCOUNT_TRADE_EXPERT={'SI' if hb['account_trade_expert'] else 'NO'}",
        f"TERMINAL_CONNECTED={'SI' if hb['terminal_connected'] else 'NO'}",
        f"TERMINAL_TRADE_ALLOWED={'SI' if hb['terminal_trade_allowed'] else 'NO'}",
        f"TRADEAPI_DISABLED={'SI' if hb['tradeapi_disabled'] else 'NO'}",
        f"PERMISSION_CONCLUSION={hb['permission_conclusion'] or '---'}",
        f"ACCION={hb['action_required'] or '---'}",
        f"ULTIMA_DECISION={hb['last_decision'] or '---'}",
        f"OPERACION_ABIERTA={operacion_abierta}",
        f"SEGURO_APAGAR_PC={seguro_apagar}",
        f"ULTIMA_ACTUALIZACION_ARG={datetime.now(AR).strftime('%H:%M:%S')}",
        f"ULTIMA_ACTUALIZACION_NY={datetime.now(NY).strftime('%H:%M:%S')}"
    ]
    write_text(QUICK_STATUS_TXT, "\n".join(qs_lines))


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from phase37d_live_news_api_adapter import news_gate_status
        from phase37d_manipulante_live_signal_engine import evaluate_live_signal

        account = account_gate()
        symbol = detect_symbol()
        session_gate = time_gate(symbol)
        news = news_gate_status(force_fetch=True)
        lot = lot_gate_10k(symbol, account)
        safety = order_send_safety()
        confirmation = confirmation_file_status()
        order_readiness = order_send_readiness_audit(symbol, account, risk=args.risk)
        
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
            "api_live_news_gate": news.get("gate"),
            "data_quality_gate": symbol.get("state"),
            "time_gate": session_gate.get("state"),
            "lot_gate": lot.get("state"),
            "signal_engine_gate": signal.get("state"),
            "order_router_safety": safety.get("state"),
            "autotrading_gate": order_readiness.get("state"),
            "terminal_trade_allowed": order_readiness.get("terminal_trade_allowed"),
            "tradeapi_disabled": order_readiness.get("tradeapi_disabled"),
            "order_check": "PASS" if order_readiness.get("order_check_pass") else "FAIL",
            "order_check_retcode": order_readiness.get("order_check_retcode"),
            "confirmation_file": confirmation.get("valid"),
            "lifecycle_state": session_state,
        }
        
        final_decision = "DRY_RUN_NO_SIGNAL" if args.dry_run else "NO_TRADE"
        reason = signal.get("signal_status", "NO_SIGNAL")
        if order_readiness.get("state") == "EMERGENCY_ABORT_REAL_MONEY_DETECTED":
            final_decision = "EMERGENCY_ABORT_REAL_MONEY_DETECTED"
            reason = "REAL_OR_EXNESS_DETECTED"
        elif order_readiness.get("state") == "BLOCKED_AUTOTRADING_DISABLED":
            final_decision = "NO_TRADE_AUTOTRADING_DISABLED"
            reason = "AUTOTRADING_DESHABILITADO"
        elif order_readiness.get("state") not in {"ORDER_CHECK_PASS_NO_ORDER_SENT"}:
            final_decision = "NO_TRADE_ORDER_CHECK_FAILED"
            reason = str(order_readiness.get("state") or "ORDER_CHECK_FAILED")
        
        # Lifecycle Logic Overrides
        if final_decision in {
            "EMERGENCY_ABORT_REAL_MONEY_DETECTED",
            "NO_TRADE_AUTOTRADING_DISABLED",
            "NO_TRADE_ORDER_CHECK_FAILED",
        }:
            pass
        elif session_state == "BEFORE_SESSION_WAIT":
            final_decision = "WAITING_FOR_SESSION"
            reason = "Pre-07:00 NY"
        elif not can_open and pos_info["state"] == "FLAT":
            final_decision = "NO_NEW_TRADES_AFTER_CUTOFF"
            reason = "Trading window closed"
        elif session_state == "WEEKEND_BLOCK":
            final_decision = "NO_TRADE_WEEKEND"
            reason = "Weekend"

        # Trading Checks
        no_check_decisions = {
            "WAITING_FOR_SESSION",
            "NO_NEW_TRADES_AFTER_CUTOFF",
            "NO_TRADE_WEEKEND",
            "EMERGENCY_ABORT_REAL_MONEY_DETECTED",
            "NO_TRADE_AUTOTRADING_DISABLED",
            "NO_TRADE_ORDER_CHECK_FAILED",
        }
        if final_decision not in no_check_decisions:
            checks = [
                (account.get("state") == "FTMO_DEMO_TRIAL_CONFIRMED", "NO_TRADE_ACCOUNT"),
                (news.get("gate") == "ALLOW", "NO_TRADE_NEWS_BLOCK"),
                (symbol.get("state") == "ALLOW", "NO_TRADE_DATA"),
                (session_gate.get("state") == "ALLOW", "NO_TRADE_TIME"),
                (lot.get("state") == "ALLOW", "NO_TRADE_LOT"),
                (signal.get("state") == "MANIPULANTE_SIGNAL_SYNC_OK", "NO_TRADE_SIGNAL"),
                (safety.get("state") == "PASS", "NO_TRADE_ORDER_ROUTER"),
                (confirmation.get("valid") is True, "CONFIRMATION_MISSING"),
            ]
            for ok, fail_reason in checks:
                if not ok:
                    final_decision = fail_reason
                    reason = fail_reason
                    break
            
            if final_decision == "ALLOW" and session_state != "SESSION_ACTIVE":
                final_decision = "NO_TRADE_OUTSIDE_ENTRY_WINDOW"
                reason = f"State {session_state} forbids entries"

        # Phase 37X-C Forced Close & Retries
        safe_close_res = None
        if not args.dry_run and lifecycle.should_force_safe_close() and pos_info["state"] != "FLAT":
            print(f"[CRITICAL] Executing Safe Close Retry Logic: {session_state}")
            safe_close_res = safe_close.execute_safe_close(symbol.get("symbol", "EURUSD"), max_attempts=5)
            final_decision = "FORCED_CLOSE_ATTEMPTED"
            reason = safe_close_res.get("status")
            pos_info = pos_state.get_position_state(symbol.get("symbol", "EURUSD"))

        # Verify Flat Windows
        flat_confirmed_1950 = False
        flat_confirmed_1955 = False
        if session_state == "VERIFY_FLAT_FIRST" and pos_info["state"] == "FLAT":
            flat_confirmed_1950 = True
        if session_state == "VERIFY_FLAT_SECOND" and pos_info["state"] == "FLAT":
            flat_confirmed_1955 = True

        shutdown_allowed = False
        critical_pc = False
        manual_req = False
        if lifecycle.should_daily_shutdown():
            if pos_info["state"] == "FLAT":
                shutdown_allowed = True
            else:
                critical_pc = True
                manual_req = True
                final_decision = "CRITICAL_MANUAL_INTERVENTION_REQUIRED"
                reason = "Position still open at deadline"

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
            "order_readiness": order_readiness,
            "shutdown_allowed": shutdown_allowed,
            "critical_do_not_turn_off_pc": critical_pc,
            "manual_intervention_required": manual_req,
            "safe_close_result": safe_close_res,
            "flat_confirmed_1950": flat_confirmed_1950,
            "flat_confirmed_1955": flat_confirmed_1955,
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
        return {"final_decision": "ERROR", "reason": str(exc), "order_sent": False}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MANIPULANTE FTMO trial bot runner")
    parser.add_argument("--ftmo-trial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--once", action="store_true", default=False)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--i-understand-demo-automation", action="store_true", default=True)
    parser.add_argument("--no-real", action="store_true", default=True)
    return parser


def safe_dumps(obj: Any) -> str:
    def default(o: Any) -> str:
        if hasattr(o, "isoformat"): return o.isoformat()
        if hasattr(o, "item"): return o.item()
        return str(o)
    return json.dumps(obj, default=default, indent=2, ensure_ascii=True)


def _other_runner_pids() -> list[int]:
    try:
        from phase37ze_quick_status_panel import find_runner_processes

        current_pid = os.getpid()
        return [proc.pid for proc in find_runner_processes() if proc.pid != current_pid]
    except Exception:
        return []


def _pid_is_running(pid: int) -> bool:
    """Check if a PID is active and belongs to a python process (if possible)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            # More robust check on Windows using tasklist or similar
            # OpenProcess can be misleading if PID is reused or process is zombie
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                text=True, stderr=subprocess.STDOUT
            )
            return str(pid) in output and "python" in output.lower()
        except Exception:
            # Fallback to ctypes if tasklist fails
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
            except:
                pass
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock() -> bool:
    """Try to acquire runner.lock. Handles stale locks automatically."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    other_pids = _other_runner_pids()
    if other_pids:
        print(f"[LOCK] Detectado otro runner activo con PID: {other_pids}")
        return False
        
    if LOCK_FILE.exists():
        try:
            content = LOCK_FILE.read_text().strip()
            if content:
                pid = int(content)
                if _pid_is_running(pid):
                    print(f"[LOCK] El archivo runner.lock indica un PID activo: {pid}")
                    return False
                else:
                    print(f"[LOCK] Limpiando lock viejo/stale (PID {pid} no existe).")
            else:
                print("[LOCK] Limpiando lock vacio.")
        except Exception as e:
            print(f"[LOCK] Error leyendo lock (limpiando): {e}")
            
        try:
            LOCK_FILE.unlink()
        except Exception as e:
            print(f"[LOCK] Error eliminando lock stale: {e}")
            return False

    try:
        LOCK_FILE.write_text(str(os.getpid()))
        return True
    except Exception as e:
        print(f"[LOCK] Error creando nuevo lock: {e}")
        return False


def release_lock():
    if LOCK_FILE.exists():
        try: LOCK_FILE.unlink()
        except: pass


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    if not args.dry_run and not acquire_lock():
        print("DUPLICATE_RUNNER_DETECTION. Aborting.")
        sys.exit(1)

    try:
        if args.once or args.dry_run:
            result = run_once(args)
            print(safe_dumps(result))
            return result
        
        print(f"MANIPULANTE FTMO TRIAL AUTO-RUNNER STARTED [PID {os.getpid()}]")
        last: dict[str, Any] = {}
        while not STOP_FILE.exists():
            last = run_once(args)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Decision: {last.get('final_decision')} | Session: {last.get('session_state')}")
            
            if last.get("shutdown_allowed"):
                print("DAILY_AUTO_SHUTDOWN REACHED and FLAT. Shutting down.")
                break
            
            time.sleep(max(10, args.interval_seconds))
        
        print("Runner finished safely.")
        return last or {"final_decision": "STOP_BOT_ACTIVE"}
    finally:
        if not args.dry_run: release_lock()


if __name__ == "__main__":
    main()
