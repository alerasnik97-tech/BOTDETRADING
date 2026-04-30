from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
SRC = ROOT / "BOT_V2_DAYTIME_LAB" / "src"

import sys

sys.path.insert(0, str(SRC))

import phase37_ftmo_trial_support as support
import phase37ze_quick_status_panel as panel

OUT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase42c_start_always_safe_idempotent"
START_BAT = ROOT / "MANIPULANTE" / "13_FTMO_TRIAL_AUTOMATION" / "START_FTMO_TRIAL_AUTO.bat"


ACCOUNT_DEMO = {
    "ftmo_demo_trial_confirmed": True,
    "company": "FTMO Global Markets Ltd",
    "server": "FTMO-Demo",
    "name": "FTMO Trial",
    "trade_mode_label": "DEMO",
    "state": "FTMO_DEMO_TRIAL_CONFIRMED",
    "terminal_trade_allowed": True,
    "reason": "ok",
}
POSITION_FLAT = {"position_open": False, "state": "FLAT_CONFIRMED", "positions_total": 0, "reason": "flat"}
POSITION_OPEN = {"position_open": True, "state": "POSITION_OPEN", "positions_total": 1, "reason": "open"}
RUNNER = [{"pid": 4242, "name": "python.exe", "command_line": "python phase37_ftmo_trial_bot_runner.py"}]


def check(name: str, condition: bool, details: dict | None = None) -> dict:
    return {
        "name": name,
        "pass": bool(condition),
        "details": details or {},
    }


def main() -> int:
    tests: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stop_file = tmp_path / "STOP_BOT.txt"
        stop_file.write_text("STOP\n", encoding="ascii")
        preflight = support.start_safety_preflight(
            stop_file=stop_file,
            runners=[],
            account_status=ACCOUNT_DEMO,
            position_status=POSITION_FLAT,
        )
        if preflight["can_clear_stop_bot"]:
            stop_file.unlink()
        tests.append(
            check(
                "START con STOP_BOT activo y sin posicion permite limpiar e iniciar",
                preflight["decision"] == "START_ALLOWED"
                and preflight["can_start"] is True
                and preflight["can_clear_stop_bot"] is True
                and not stop_file.exists(),
                preflight,
            )
        )

    already = support.start_safety_preflight(
        runners=RUNNER,
        account_status=ACCOUNT_DEMO,
        position_status=POSITION_FLAT,
    )
    tests.append(
        check(
            "START con bot activo no duplica",
            already["decision"] == "ALREADY_RUNNING" and already["runner_count"] == 1,
            already,
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        lock_dir = Path(tmp) / "start.lock"
        first = support.acquire_start_lock(lock_dir, runners=[])
        second = support.acquire_start_lock(lock_dir, runners=[])
        third = support.acquire_start_lock(lock_dir, runners=[])
        support.release_start_lock(lock_dir)
        tests.append(
            check(
                "START tocado 3 veces conserva un solo permiso de arranque",
                first["acquired"] is True and second["acquired"] is False and third["acquired"] is False,
                {"first": first, "second": second, "third": third},
            )
        )

    old_live_position = panel._live_position_open
    old_stop_bot = panel.STOP_BOT
    with tempfile.TemporaryDirectory() as tmp:
        stop_status = Path(tmp) / "STOP_BOT.txt"
        stop_status.write_text("STOP\n", encoding="ascii")
        try:
            panel._live_position_open = lambda: None
            panel.STOP_BOT = stop_status
            status = panel.build_status(
                runners=[],
                qs={"CUENTA": "FTMO-Demo", "OPERACION_ABIERTA": "NO"},
                hb={},
                last_decision_row={},
                mt5="CERRADO",
            )
        finally:
            panel._live_position_open = old_live_position
            panel.STOP_BOT = old_stop_bot
    tests.append(
        check(
            "STATUS con STOP_BOT activo muestra BOT DETENIDO",
            status["ESTADO_GENERAL"] == "BOT DETENIDO"
            and status["BOT"] == "APAGADO"
            and status["MOTIVO"] == "STOP_BOT ACTIVO",
            status,
        )
    )

    real = support.start_safety_preflight(
        runners=[],
        account_status={**ACCOUNT_DEMO, "trade_mode_label": "REAL", "state": "BLOCKED_REAL_ACCOUNT_DETECTED"},
        position_status=POSITION_FLAT,
    )
    tests.append(
        check(
            "START con cuenta real simulada hace emergency abort",
            real["decision"] == "EMERGENCY_ABORT_REAL_OR_EXNESS" and real["real_detected"] is True,
            real,
        )
    )

    exness = support.start_safety_preflight(
        runners=[],
        account_status={**ACCOUNT_DEMO, "company": "Exness", "server": "Exness-Demo"},
        position_status=POSITION_FLAT,
    )
    tests.append(
        check(
            "START con Exness simulado hace emergency abort",
            exness["decision"] == "EMERGENCY_ABORT_REAL_OR_EXNESS" and exness["exness_detected"] is True,
            exness,
        )
    )

    pos_open = support.start_safety_preflight(
        runners=[],
        account_status=ACCOUNT_DEMO,
        position_status=POSITION_OPEN,
    )
    tests.append(
        check(
            "START con posicion abierta no limpia ni reinicia",
            pos_open["decision"] == "BLOCKED_POSITION_OPEN"
            and pos_open["can_start"] is False
            and pos_open["can_clear_stop_bot"] is False,
            pos_open,
        )
    )

    bat_text = START_BAT.read_text(encoding="utf-8", errors="ignore")
    security = {
        "order_sent": False,
        "strategy_modified": False,
        "real_touched": False,
        "exness_touched": False,
        "runner_invocations_in_start_bat": bat_text.count("phase37_ftmo_trial_bot_runner.py"),
        "contains_no_real_flag": "--no-real" in bat_text,
        "contains_order_send_call": "order_send(" in bat_text,
    }
    tests.append(
        check(
            "Seguridad estatica de START",
            security["runner_invocations_in_start_bat"] == 1
            and security["contains_no_real_flag"] is True
            and security["contains_order_send_call"] is False,
            security,
        )
    )

    total = len(tests)
    passed = sum(1 for item in tests if item["pass"])
    failed = total - passed
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE42C_START_ALWAYS_SAFE_IDEMPOTENT",
        "total": total,
        "pass": passed,
        "fail": failed,
        "tests": tests,
        "security": security,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# Phase42C Start Idempotence Tests",
        "",
        f"- total: {total}",
        f"- pass: {passed}",
        f"- fail: {failed}",
        "",
        "## Results",
    ]
    for item in tests:
        state = "PASS" if item["pass"] else "FAIL"
        lines.append(f"- {state}: {item['name']}")
    (OUT / "test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"total": total, "pass": passed, "fail": failed}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
