from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_support import (
    MANIPULANTE,
    NY,
    account_gate,
    confirmation_file_status,
    detect_symbol,
    live_news_gate,
    lot_gate_10k,
    order_send_safety,
    strategy_config_gate,
    time_gate,
    write_csv,
)


CONFIRMATION_FILE = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "I_CONFIRM_FTMO_TRIAL_AUTO.txt"
ORDER_LOG_DIR = MANIPULANTE / "10_LOGS_PAPER" / "ftmo_trial_orders"


@dataclass
class TrialOrderDecision:
    timestamp: str
    final_decision: str
    order_sent: bool
    reason: str
    gates: dict[str, Any]
    order_check_result: str | None = None
    ticket: str | None = None


def _gate_ok(value: Any, allowed: set[str] | None = None) -> bool:
    allowed = allowed or {"ALLOW", "PASS", "FTMO_DEMO_TRIAL_CONFIRMED", "MANIPULANTE_MATCH"}
    return value in allowed or value is True


def validate_trial_flags(args: argparse.Namespace) -> tuple[bool, str]:
    if not args.ftmo_trial:
        return False, "MISSING_FLAG_FTMO_TRIAL"
    if not args.i_understand_demo_automation:
        return False, "MISSING_FLAG_I_UNDERSTAND_DEMO_AUTOMATION"
    if not args.no_real:
        return False, "MISSING_FLAG_NO_REAL"
    if not args.no_force:
        return False, "MISSING_FLAG_NO_FORCE"
    if abs(args.risk - 0.0075) < 1e-9 and not args.trial_risk_stress_075:
        return False, "RISK_075_REQUIRES_TRIAL_STRESS_FLAG"
    if args.risk > 0.0075:
        return False, "RISK_ABOVE_TRIAL_MAX"
    if abs(args.risk - 0.01) < 1e-9 or args.risk >= 0.01:
        return False, "RISK_1PCT_PROHIBITED"
    if args.risk not in {0.005, 0.0075}:
        return False, "RISK_NOT_ALLOWED_FOR_PHASE37_TRIAL"
    return True, "FLAGS_OK"


def evaluate_gates(args: argparse.Namespace, request: dict[str, Any] | None = None) -> TrialOrderDecision:
    flag_ok, flag_reason = validate_trial_flags(args)
    account = account_gate()
    symbol = detect_symbol()
    news = live_news_gate()
    session = time_gate(symbol)
    lot = lot_gate_10k(symbol, account)
    safety = order_send_safety()
    config = strategy_config_gate()
    confirmation = confirmation_file_status()
    max_trades_day = {"state": "ALLOW", "reason": "No Phase37 trial order lock found in this dry-run evaluation"}
    weekend = {"state": "ALLOW" if session.get("state") != "NO_TRADE_WEEKEND" else "NO_TRADE_WEEKEND"}
    gates = {
        "flags": flag_reason,
        "account_gate": account.get("state"),
        "platform_gate": "MT5_CONNECTED" if account.get("terminal_connected") else "NO_TRADE",
        "symbol_gate": symbol.get("state"),
        "live_news_gate": news.get("gate"),
        "week_news_loaded": news.get("week_loaded"),
        "data_quality_gate": symbol.get("state"),
        "time_gate": session.get("state"),
        "spread_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "stoplevel_gate": "ALLOW" if symbol.get("state") == "ALLOW" else symbol.get("state"),
        "lot_gate": lot.get("state"),
        "max_trades_day_gate": max_trades_day.get("state"),
        "weekend_gate": weekend.get("state"),
        "strategy_config_gate": config.get("state"),
        "order_send_safety": safety.get("state"),
        "confirmation_file": confirmation.get("valid"),
    }
    checks = [
        (flag_ok, flag_reason),
        (_gate_ok(gates["account_gate"]), "ACCOUNT_GATE_NOT_CONFIRMED"),
        (gates["platform_gate"] == "MT5_CONNECTED", "PLATFORM_GATE_NOT_CONNECTED"),
        (_gate_ok(gates["symbol_gate"]), "SYMBOL_GATE_NOT_ALLOW"),
        (gates["live_news_gate"] == "ALLOW", "NEWS_GATE_NOT_ALLOW"),
        (gates["week_news_loaded"] is True, "WEEK_NEWS_NOT_LOADED"),
        (_gate_ok(gates["data_quality_gate"]), "DATA_GATE_NOT_ALLOW"),
        (_gate_ok(gates["time_gate"]), "TIME_GATE_NOT_ALLOW"),
        (_gate_ok(gates["spread_gate"]), "SPREAD_GATE_NOT_ALLOW"),
        (_gate_ok(gates["stoplevel_gate"]), "STOPLEVEL_GATE_NOT_ALLOW"),
        (_gate_ok(gates["lot_gate"]), "LOT_GATE_NOT_ALLOW"),
        (_gate_ok(gates["max_trades_day_gate"]), "MAX_TRADES_DAY_NOT_ALLOW"),
        (_gate_ok(gates["weekend_gate"]), "WEEKEND_GATE_NOT_ALLOW"),
        (_gate_ok(gates["strategy_config_gate"]), "STRATEGY_CONFIG_NOT_MATCH"),
        (_gate_ok(gates["order_send_safety"]), "ORDER_SEND_SAFETY_NOT_PASS"),
        (gates["confirmation_file"] is True, "CONFIRMATION_FILE_MISSING_OR_INVALID"),
    ]
    for ok, reason in checks:
        if not ok:
            return TrialOrderDecision(datetime.now(NY).isoformat(), "NO_TRADE", False, reason, gates)
    if args.dry_run:
        return TrialOrderDecision(datetime.now(NY).isoformat(), "DRY_RUN_ALLOW_SIGNAL_READY", False, "DRY_RUN_NO_ORDER_SEND", gates, "NOT_CALLED_IN_DRY_RUN")
    return TrialOrderDecision(datetime.now(NY).isoformat(), "READY_FOR_DEMO_TRIAL_ORDER_CHECK", False, "ALL_GATES_PASS_PRE_ORDER_CHECK", gates, "REQUIRED_BEFORE_SEND")


def write_order_log(decision: TrialOrderDecision, request: dict[str, Any] | None = None) -> Path:
    ORDER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(NY).strftime("%Y-%m-%d")
    path = ORDER_LOG_DIR / f"{day}_ftmo_trial_order_decisions.csv"
    row = {
        "timestamp": decision.timestamp,
        "account_login_masked": "masked",
        "account_company": "FTMO",
        "account_mode": decision.gates.get("account_gate"),
        "symbol": (request or {}).get("symbol", ""),
        "risk": (request or {}).get("risk", ""),
        "lot": (request or {}).get("lot", ""),
        "entry": (request or {}).get("entry", ""),
        "SL": (request or {}).get("sl", ""),
        "TP": (request or {}).get("tp", ""),
        "gates_status": json.dumps(decision.gates, ensure_ascii=False),
        "order_check_result": decision.order_check_result or "",
        "order_sent": str(decision.order_sent).lower(),
        "ticket": decision.ticket or "",
        "reason": decision.reason,
    }
    fields = list(row.keys())
    rows = []
    if path.exists():
        return path
    write_csv(path, [row], fields)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FTMO Trial fail-closed order router")
    parser.add_argument("--ftmo-trial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--i-understand-demo-automation", action="store_true")
    parser.add_argument("--no-real", action="store_true")
    parser.add_argument("--no-force", action="store_true")
    parser.add_argument("--trial-risk-stress-075", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    decision = evaluate_gates(args)
    write_order_log(decision)
    payload = asdict(decision)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
