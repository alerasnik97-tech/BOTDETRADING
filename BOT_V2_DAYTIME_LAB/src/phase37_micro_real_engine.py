from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIRMATION_FILE = ROOT / "MANIPULANTE" / "12_MICRO_REAL_READINESS" / "I_CONFIRM_MICRO_REAL.txt"
OUT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase36r_37a_micro_real_gate" / "micro_real_engine"

REQUIRED_CONFIRMATION = [
    "I UNDERSTAND THIS CAN LOSE REAL MONEY",
    "I CONFIRM MICRO REAL ONLY",
    "RISK_MAX=0.25",
    "NO_AUTOTRADING_BLIND",
    "ONE_TRADE_ONLY",
]


def confirmation_file_gate(path: Path = CONFIRMATION_FILE) -> dict[str, Any]:
    if not path.exists():
        return {"state": "NO_TRADE", "present": False, "reason": "confirmation file absent"}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [line for line in REQUIRED_CONFIRMATION if line not in lines]
    if missing:
        return {"state": "NO_TRADE", "present": True, "reason": "confirmation file invalid", "missing": missing}
    return {"state": "ALLOW", "present": True, "reason": "confirmation file valid"}


def build_micro_real_decision(args: argparse.Namespace | None = None, gates: dict[str, Any] | None = None) -> dict[str, Any]:
    args = args or argparse.Namespace(micro_real=False, risk=1.0, i_understand_real_risk=False, no_force=False)
    gates = gates or {}
    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "PHASE37_MICRO_REAL_ENGINE",
        "blocked_by_default": True,
        "order_send_enabled": False,
        "order_check_required": True,
        "confirmation_file_required": True,
        "order_sent": False,
        "final_decision": "NO_TRADE",
        "reason": "",
        "checks": {
            "flag_micro_real": bool(getattr(args, "micro_real", False)),
            "flag_i_understand_real_risk": bool(getattr(args, "i_understand_real_risk", False)),
            "flag_no_force": bool(getattr(args, "no_force", False)),
            "risk": float(getattr(args, "risk", 1.0)),
            "confirmation_file": confirmation_file_gate(),
            "gates": gates,
        },
    }
    if not decision["checks"]["flag_micro_real"]:
        decision["reason"] = "missing --micro-real"
        return decision
    if not decision["checks"]["flag_i_understand_real_risk"]:
        decision["reason"] = "missing --i-understand-real-risk"
        return decision
    if not decision["checks"]["flag_no_force"]:
        decision["reason"] = "missing --no-force"
        return decision
    if decision["checks"]["risk"] not in {0.001, 0.0025}:
        decision["reason"] = "risk must be exactly 0.001 or 0.0025"
        return decision
    if decision["checks"]["confirmation_file"]["state"] != "ALLOW":
        decision["reason"] = decision["checks"]["confirmation_file"]["reason"]
        return decision
    required = [
        "news_gate", "week_news_loaded", "data_gate", "time_gate", "symbol_gate",
        "spread_gate", "stoplevel_gate", "lot_gate", "max_trades_day_gate",
        "weekend_gate", "order_send_safety", "order_check",
    ]
    for key in required:
        if gates.get(key) not in {"ALLOW", "PASS", True}:
            decision["reason"] = f"gate not allow: {key}"
            return decision
    decision["final_decision"] = "READY_WITH_WARNINGS_NEEDS_USER_FINAL_CONFIRMATION"
    decision["reason"] = "all software gates passed, but this phase never sends orders automatically"
    return decision


def write_outputs(decision: dict[str, Any] | None = None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    decision = decision or build_micro_real_decision()
    (OUT / "phase37_micro_real_engine_status.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    md = [
        "# Phase37 Micro Real Engine",
        "",
        f"- created: yes",
        f"- blocked_by_default: {decision['blocked_by_default']}",
        f"- confirmation_file_required: {decision['confirmation_file_required']}",
        f"- order_check_required: {decision['order_check_required']}",
        f"- final_decision: {decision['final_decision']}",
        f"- reason: {decision['reason']}",
        "",
        "This engine does not send real orders in Phase36R/37A.",
    ]
    (OUT / "phase37_micro_real_engine_status.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return decision


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-real", action="store_true")
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--i-understand-real-risk", action="store_true")
    parser.add_argument("--no-force", action="store_true")
    decision = write_outputs(build_micro_real_decision(parser.parse_args()))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
