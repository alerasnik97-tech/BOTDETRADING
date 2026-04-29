from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_order_router import evaluate_gates
from phase37_ftmo_trial_support import MANIPULANTE, NY, OUT, account_gate, confirmation_file_status, detect_symbol, lot_gate_10k, order_send_safety, signal_sync, time_gate, write_json, write_text


STOP_FILE = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"
DECISION_LOG = MANIPULANTE / "10_LOGS_PAPER" / "ftmo_trial_bot" / "decisions.csv"


def append_decision(row: dict[str, Any]) -> None:
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = DECISION_LOG.exists()
    fields = [
        "timestamp",
        "final_decision",
        "reason",
        "signal_status",
        "order_sent",
        "gates_status",
    ]
    with DECISION_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    api_mode = False
    try:
        from phase37d_live_news_api_adapter import news_gate_status
        from phase37d_manipulante_live_signal_engine import evaluate_live_signal

        api_mode = True
        account = account_gate()
        symbol = detect_symbol()
        session = time_gate(symbol)
        news = news_gate_status(force_fetch=True)
        lot = lot_gate_10k(symbol, account)
        safety = order_send_safety()
        confirmation = confirmation_file_status()
        signal = evaluate_live_signal(
            news_gate=news.get("gate", "NO_TRADE"),
            data_gate=symbol.get("state", "NO_TRADE"),
            time_state=session.get("state"),
        )
        gates = {
            "account_gate": account.get("state"),
            "real_money_gate": "REAL_BLOCKED",
            "mt5_connection_gate": "ALLOW" if account.get("terminal_connected") else "NO_TRADE",
            "api_live_news_gate": news.get("gate"),
            "week_news_loaded": news.get("week_news_loaded"),
            "data_quality_gate": symbol.get("state"),
            "time_gate": session.get("state"),
            "symbol_gate": symbol.get("state"),
            "spread_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
            "stoplevel_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
            "lot_gate": lot.get("state"),
            "signal_engine_gate": signal.get("state"),
            "max_trades_day_gate": "ALLOW",
            "weekend_gate": "ALLOW" if session.get("state") != "NO_TRADE_WEEKEND" else "NO_TRADE_WEEKEND",
            "order_router_safety": safety.get("state"),
            "confirmation_file": confirmation.get("valid"),
        }
        final_decision = "DRY_RUN_NO_SIGNAL" if args.dry_run else "NO_TRADE"
        reason = signal.get("signal_status", "NO_SIGNAL")
        checks = [
            (account.get("state") == "FTMO_DEMO_TRIAL_CONFIRMED", "NO_TRADE_ACCOUNT"),
            (news.get("gate") == "ALLOW", "NO_TRADE_NEWS_BLOCK"),
            (news.get("week_news_loaded") is True, "NO_TRADE_WEEK_NEWS_NOT_LOADED"),
            (symbol.get("state") == "ALLOW", "NO_TRADE_DATA_OR_SYMBOL"),
            (session.get("state") == "ALLOW", "NO_TRADE_TIME"),
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
        decision = type("Decision", (), {"gates": gates})()
        sync = signal
    except Exception:
        decision = evaluate_gates(args)
        sync = signal_sync()
        final_decision = decision.final_decision
        reason = decision.reason
        if sync["state"] != "MANIPULANTE_SIGNAL_SYNC_OK":
            final_decision = "NO_TRADE_SIGNAL_NOT_READY"
            reason = sync["state"]
    if STOP_FILE.exists():
        final_decision = "STOP_BOT_ACTIVE"
        reason = "STOP_BOT.txt present"
    row = {
        "timestamp": datetime.now(NY).isoformat(),
        "final_decision": final_decision,
        "reason": reason,
        "signal_status": sync["state"],
        "order_sent": False,
        "gates_status": json.dumps(decision.gates, ensure_ascii=False),
    }
    append_decision(row)
    return {
        "timestamp": row["timestamp"],
        "dry_run": args.dry_run,
        "final_decision": final_decision,
        "reason": reason,
        "order_sent": False,
        "signal_sync": sync,
        "gates": decision.gates,
        "api_news_primary": api_mode,
    }


def write_dry_run_outputs(result: dict[str, Any]) -> None:
    out_dir = OUT / "dry_run"
    write_json(out_dir / "phase37_dry_run.json", result)
    csv_path = out_dir / "phase37_dry_run_decisions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "decision", "reason", "order_sent"])
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": result["timestamp"],
                "decision": result["final_decision"],
                "reason": result["reason"],
                "order_sent": result["order_sent"],
            }
        )
    write_text(
        out_dir / "phase37_dry_run.md",
        f"""
# Phase37 FTMO Trial Dry-Run

- executed: true
- decision: {result['final_decision']}
- reason: {result['reason']}
- order_sent: {result['order_sent']}

Dry-run never sends orders.
""",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MANIPULANTE FTMO trial bot runner")
    parser.add_argument("--ftmo-trial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--i-understand-demo-automation", action="store_true", default=True)
    parser.add_argument("--no-real", action="store_true", default=True)
    parser.add_argument("--no-force", action="store_true", default=True)
    parser.add_argument("--trial-risk-stress-075", action="store_true")
    parser.add_argument("--once", action="store_true", default=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.once or args.dry_run:
        result = run_once(args)
        write_dry_run_outputs(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    last: dict[str, Any] = {}
    while not STOP_FILE.exists():
        last = run_once(args)
        time.sleep(max(10, args.interval_seconds))
    if not last:
        last = {"final_decision": "STOP_BOT_ACTIVE", "order_sent": False, "reason": "STOP_BOT.txt present before loop"}
    print(json.dumps(last, indent=2, ensure_ascii=False))
    return last


if __name__ == "__main__":
    main()
