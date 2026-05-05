from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # pragma: no cover - dry-run environments may not have MT5.
    mt5 = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_FILE = ROOT / "MANIPULANTE" / "12_MICRO_REAL_READINESS" / "I_CONFIRM_MICRO_REAL.txt"
ORDER_LOCK_DIR = ROOT / "MANIPULANTE" / "10_LOGS_PAPER" / "order_sent_locks"
NY = ZoneInfo("America/New_York")

REQUIRED_CONFIRMATION_LINES = [
    "I UNDERSTAND THIS CAN LOSE REAL MONEY",
    "I CONFIRM MICRO REAL ONLY",
    "RISK_MAX=0.25",
    "NO_AUTOTRADING_BLIND",
    "ONE_TRADE_ONLY",
]

REQUIRED_ALLOW_GATES = [
    "news_gate",
    "data_gate",
    "time_gate",
    "symbol_gate",
    "spread_gate",
    "stoplevel_gate",
    "lot_gate",
    "max_trades_day_gate",
    "weekend_gate",
]


@dataclass
class GuardResult:
    final_decision: str
    order_sent: bool
    reason: str
    checks: dict[str, Any]
    request: dict[str, Any] | None = None
    mt5_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _today_lock_file(symbol: str) -> Path:
    ORDER_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(NY).strftime("%Y-%m-%d")
    return ORDER_LOCK_DIR / f"{day}_{symbol}_order_sent.lock"


def confirmation_file_valid(path: Path = CONFIRMATION_FILE) -> tuple[bool, str]:
    if not path.exists():
        return False, "CONFIRMATION_FILE_MISSING"
    try:
        content = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        return False, f"CONFIRMATION_FILE_READ_ERROR: {exc}"
    missing = [line for line in REQUIRED_CONFIRMATION_LINES if line not in content]
    if missing:
        return False, "CONFIRMATION_FILE_INVALID: " + ",".join(missing)
    return True, "CONFIRMATION_FILE_VALID"


def _gate_is_allow(value: Any) -> bool:
    return value in {True, "TRUE", "ALLOW", "PASS"}


def validate_micro_real_gates(
    request: dict[str, Any],
    *,
    mode: str,
    micro_real_flag: bool,
    understand_real_risk_flag: bool,
    no_force_flag: bool,
    risk: float,
    gates: dict[str, Any],
    confirmation_path: Path = CONFIRMATION_FILE,
    order_check_result: Any = None,
) -> GuardResult:
    checks: dict[str, Any] = {
        "mode": mode,
        "micro_real_flag": micro_real_flag,
        "understand_real_risk_flag": understand_real_risk_flag,
        "no_force_flag": no_force_flag,
        "risk": risk,
        "confirmation_file": None,
        "required_gates": {},
        "order_check": order_check_result,
        "sl_present": bool(request.get("sl")),
        "tp_present": bool(request.get("tp")),
        "order_sent_lock_exists": _today_lock_file(str(request.get("symbol", "EURUSD"))).exists(),
    }
    if mode != "MICRO_REAL_ONLY":
        return GuardResult("NO_TRADE", False, "MODE_NOT_MICRO_REAL_ONLY", checks, request)
    if not micro_real_flag:
        return GuardResult("NO_TRADE", False, "MISSING_FLAG_MICRO_REAL", checks, request)
    if not understand_real_risk_flag:
        return GuardResult("NO_TRADE", False, "MISSING_FLAG_I_UNDERSTAND_REAL_RISK", checks, request)
    if not no_force_flag:
        return GuardResult("NO_TRADE", False, "MISSING_FLAG_NO_FORCE", checks, request)
    if risk > 0.0025:
        return GuardResult("NO_TRADE", False, "RISK_ABOVE_0_25_PERCENT", checks, request)
    confirmation_ok, confirmation_reason = confirmation_file_valid(confirmation_path)
    checks["confirmation_file"] = confirmation_reason
    if not confirmation_ok:
        return GuardResult("NO_TRADE", False, confirmation_reason, checks, request)
    if not checks["sl_present"]:
        return GuardResult("NO_TRADE", False, "SL_MISSING", checks, request)
    if not checks["tp_present"]:
        return GuardResult("NO_TRADE", False, "TP_MISSING", checks, request)
    if checks["order_sent_lock_exists"]:
        return GuardResult("NO_TRADE", False, "ORDER_SENT_LOCK_ALREADY_USED_TODAY", checks, request)
    for gate in REQUIRED_ALLOW_GATES:
        value = gates.get(gate)
        checks["required_gates"][gate] = value
        if not _gate_is_allow(value):
            return GuardResult("NO_TRADE", False, f"GATE_NOT_ALLOW:{gate}", checks, request)
    if not _gate_is_allow(order_check_result):
        return GuardResult("NO_TRADE", False, "ORDER_CHECK_NOT_PASS", checks, request)
    return GuardResult("READY_TO_SEND", False, "ALL_GATES_PASS_PRE_SEND", checks, request)


def safe_order_send_guarded(
    request: dict[str, Any],
    *,
    mode: str = "DRY_RUN_ONLY",
    micro_real_flag: bool = False,
    understand_real_risk_flag: bool = False,
    no_force_flag: bool = False,
    risk: float = 1.0,
    gates: dict[str, Any] | None = None,
    confirmation_path: Path = CONFIRMATION_FILE,
    order_check_func: Callable[[dict[str, Any]], Any] | None = None,
    mt5_module: Any = None,
) -> GuardResult:
    """Single guarded wrapper around MT5 order_send.

    In dry-run/default mode this function always returns NO_TRADE and never
    calls MT5. Real sending requires all explicit flags, exact confirmation
    file, risk <= 0.25%, all gates ALLOW/PASS, SL/TP and order_check PASS.
    """
    if mode == "DRY_RUN_ONLY":
        return GuardResult("NO_TRADE", False, "DRY_RUN_ONLY_ORDER_SEND_DISABLED", {"mode": mode}, request)

    gates = gates or {}
    order_check_result = None
    if order_check_func is not None:
        try:
            order_check_result = order_check_func(request)
        except Exception as exc:
            order_check_result = f"ORDER_CHECK_EXCEPTION:{exc}"

    guard = validate_micro_real_gates(
        request,
        mode=mode,
        micro_real_flag=micro_real_flag,
        understand_real_risk_flag=understand_real_risk_flag,
        no_force_flag=no_force_flag,
        risk=risk,
        gates=gates,
        confirmation_path=confirmation_path,
        order_check_result=order_check_result,
    )
    if guard.final_decision != "READY_TO_SEND":
        return guard

    active_mt5 = mt5_module if mt5_module is not None else mt5
    if active_mt5 is None:
        return GuardResult("NO_TRADE", False, "MT5_MODULE_UNAVAILABLE", guard.checks, request)

    result = active_mt5.order_send(request)
    _today_lock_file(str(request.get("symbol", "EURUSD"))).write_text(
        json.dumps({"timestamp": datetime.now(NY).isoformat(), "symbol": request.get("symbol")}, indent=2),
        encoding="utf-8",
    )
    return GuardResult("ORDER_SENT", True, "ORDER_SEND_EXECUTED_AFTER_ALL_GATES", guard.checks, request, str(result))


class MT5OrderRouter:
    def __init__(self, symbol: str = "EURUSD", magic: int = 123456, mode: str = "DRY_RUN_ONLY") -> None:
        self.symbol = symbol
        self.magic = magic
        self.mode = mode

    def _build_request(self, action: str, volume: float, price: float | None = None, sl: float | None = None, tp: float | None = None, comment: str = "MANIPULANTE") -> dict[str, Any]:
        order_type = "BUY" if action.upper() == "BUY" else "SELL"
        return {
            "action": "TRADE_ACTION_DEAL",
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "deviation": 20,
            "magic": self.magic,
            "comment": comment,
            "type_time": "ORDER_TIME_GTC",
            "type_filling": "ORDER_FILLING_IOC",
        }

    def send_order(self, action: str, volume: float, price: float | None = None, sl: float | None = None, tp: float | None = None, comment: str = "MANIPULANTE", **kwargs: Any) -> GuardResult:
        request = self._build_request(action, volume, price, sl, tp, comment)
        return safe_order_send_guarded(request, mode=self.mode, **kwargs)

    def close_position(self, ticket: int, comment: str = "Close position", **kwargs: Any) -> GuardResult:
        request = {
            "action": "CLOSE_POSITION",
            "symbol": self.symbol,
            "position": ticket,
            "comment": comment,
            "sl": 1.0,
            "tp": 1.0,
        }
        return safe_order_send_guarded(request, mode=self.mode, **kwargs)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="MANIPULANTE fail-closed MT5 router")
    parser.add_argument("--dry-run-only", action="store_true", help="Never call order_send")
    parser.add_argument("--micro-real", action="store_true", help="Explicit micro-real flag")
    parser.add_argument("--i-understand-real-risk", action="store_true", help="Explicit risk acknowledgement")
    parser.add_argument("--no-force", action="store_true", help="Required anti-force flag")
    parser.add_argument("--risk", type=float, default=1.0)
    args = parser.parse_args()
    mode = "DRY_RUN_ONLY" if args.dry_run_only else "MICRO_REAL_ONLY"
    router = MT5OrderRouter(mode=mode)
    result = router.send_order(
        "BUY",
        0.01,
        price=1.0,
        sl=0.999,
        tp=1.0014,
        micro_real_flag=args.micro_real,
        understand_real_risk_flag=args.i_understand_real_risk,
        no_force_flag=args.no_force,
        risk=args.risk,
        gates={},
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if not result.order_sent else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
