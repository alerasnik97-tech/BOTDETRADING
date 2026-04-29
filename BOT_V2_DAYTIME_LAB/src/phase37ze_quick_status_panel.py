from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LOG_DIR = ROOT / "MANIPULANTE" / "10_LOGS_PAPER" / "ftmo_trial_bot"
QUICK_STATUS = LOG_DIR / "quick_status.txt"
HEARTBEAT = LOG_DIR / "heartbeat.json"
DECISIONS = LOG_DIR / "decisions.csv"

RUNNER_SCRIPT = "phase37_ftmo_trial_bot_runner.py"
STATUS_SCRIPT = "phase37ze_quick_status_panel.py"
VALID_STATES = {"VERDE", "AMARILLO", "ROJO", "CRITICO", "VIOLETA"}


@dataclass(frozen=True)
class RunnerProcess:
    pid: int
    name: str
    command_line: str


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_valid_runner_process(proc: dict[str, Any]) -> bool:
    name = str(proc.get("Name") or "").lower()
    command_line = str(proc.get("CommandLine") or "")
    command_lower = command_line.lower()

    if name not in {"python.exe", "pythonw.exe"}:
        return False
    if RUNNER_SCRIPT.lower() not in command_lower:
        return False

    blocked_tokens = [
        STATUS_SCRIPT.lower(),
        "status_manipulante",
        "status_ftmo_trial_auto",
        "get-ciminstance",
        "powershell",
    ]
    return not any(token in command_lower for token in blocked_tokens)


def find_runner_processes() -> list[RunnerProcess]:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,Name,CommandLine | "
            "ConvertTo-Json -Compress"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        processes = _as_list(json.loads(result.stdout))
    except Exception:
        return []

    runners: list[RunnerProcess] = []
    for proc in processes:
        if not isinstance(proc, dict) or not _is_valid_runner_process(proc):
            continue
        try:
            pid = int(proc.get("ProcessId"))
        except Exception:
            continue
        runners.append(
            RunnerProcess(
                pid=pid,
                name=str(proc.get("Name") or ""),
                command_line=str(proc.get("CommandLine") or ""),
            )
        )
    return sorted(runners, key=lambda item: item.pid)


def get_mt5_status() -> str:
    try:
        result = subprocess.run(
            'tasklist /FI "IMAGENAME eq terminal64.exe"',
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
        return "ABIERTO" if "terminal64.exe" in result.stdout.lower() else "CERRADO"
    except Exception:
        return "DESCONOCIDO"


def read_key_value(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for encoding in ("utf-8", "cp1252"):
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            return data
    else:
        return data

    for line in lines:
        if "=" not in line:
            continue
        key, value = line.strip().split("=", 1)
        data[key.strip()] = value.strip()
    return data


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def read_last_decision(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    return rows[-1] if rows else {}


def _hhmm(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "T" not in text and len(text) >= 5:
        return text[:5]
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%H:%M")
    except Exception:
        return text[:5] if len(text) >= 5 else text


def _yes_no(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "SI" if value else "NO"
    text = str(value).strip().upper()
    if text in {"SI", "YES", "TRUE", "1"}:
        return "SI"
    if text in {"NO", "FALSE", "0"}:
        return "NO"
    return None


def _account_label(qs: dict[str, str], hb: dict[str, Any]) -> str:
    server = str(hb.get("server") or "").strip()
    mode = str(hb.get("account_mode") or "").strip()
    if server or mode:
        return f"{server or 'FTMO-Demo'} / {mode or 'DEMO'}"
    cuenta = str(qs.get("CUENTA") or "").strip()
    if cuenta:
        if "demo" in cuenta.lower() or "ftmo" in cuenta.lower():
            return "FTMO-Demo / DEMO"
        return cuenta
    return "FTMO-Demo / DEMO"


def _operation_open(qs: dict[str, str], hb: dict[str, Any]) -> str:
    from_qs = _yes_no(qs.get("OPERACION_ABIERTA"))
    if from_qs:
        return from_qs
    position = str(hb.get("position_state") or "").strip().upper()
    if not position:
        return "NO"
    return "NO" if position == "FLAT" else "SI"


def _safe_to_turn_off(qs: dict[str, str], hb: dict[str, Any], operation_open: str, runner_count: int) -> str:
    from_qs = _yes_no(qs.get("SEGURO_APAGAR_PC") or qs.get("SAFE_TO_TURN_OFF_PC"))
    if from_qs:
        return from_qs
    if operation_open == "SI":
        return "NO"
    from_hb = _yes_no(hb.get("safe_to_turn_off_pc"))
    if from_hb:
        return from_hb
    return "SI" if runner_count == 0 else "NO"


def build_status(
    runners: list[RunnerProcess] | None = None,
    qs: dict[str, str] | None = None,
    hb: dict[str, Any] | None = None,
    last_decision_row: dict[str, str] | None = None,
    mt5: str | None = None,
) -> dict[str, str]:
    runners = find_runner_processes() if runners is None else runners
    qs = read_key_value(QUICK_STATUS) if qs is None else qs
    hb = read_json(HEARTBEAT) if hb is None else hb
    last_decision_row = read_last_decision(DECISIONS) if last_decision_row is None else last_decision_row
    mt5 = get_mt5_status() if mt5 is None else mt5

    runner_count = len(runners)
    pids = ", ".join(str(proc.pid) for proc in runners) if runners else "---"
    runner_state = "ACTIVO" if runner_count > 0 else "APAGADO"
    news = str(qs.get("NEWS") or hb.get("news_gate") or "---").strip() or "---"
    decision = str(
        qs.get("ULTIMA_DECISION")
        or hb.get("last_decision")
        or last_decision_row.get("final_decision")
        or "---"
    ).strip()
    operation_open = _operation_open(qs, hb)
    safe_off = _safe_to_turn_off(qs, hb, operation_open, runner_count)

    estado = str(qs.get("ESTADO_GENERAL") or "").strip().upper()
    if estado not in VALID_STATES:
        estado = "VERDE"
    if runner_count > 1:
        estado = "VIOLETA"
    elif runner_count == 0:
        estado = "ROJO"
    elif operation_open == "SI" or _yes_no(hb.get("critical_position_still_open")) == "SI":
        estado = "CRITICO"
    elif news not in {"ALLOW", "---"} or _yes_no(hb.get("manual_intervention_required")) == "SI":
        estado = "AMARILLO"

    arg_time = (
        _hhmm(qs.get("ULTIMA_ACTUALIZACION_ARG"))
        or _hhmm(hb.get("timestamp_local"))
        or _hhmm(last_decision_row.get("timestamp_local"))
        or "---"
    )
    ny_time = (
        _hhmm(qs.get("ULTIMA_ACTUALIZACION_NY"))
        or _hhmm(qs.get("LAST_UPDATE"))
        or _hhmm(hb.get("timestamp_ny"))
        or _hhmm(last_decision_row.get("timestamp_ny"))
        or "---"
    )

    return {
        "ESTADO_GENERAL": estado,
        "CUENTA": _account_label(qs, hb),
        "RUNNER": runner_state,
        "PID_RUNNER": pids,
        "MT5": mt5,
        "NEWS": news,
        "ULTIMA_DECISION": decision or "---",
        "OPERACION_ABIERTA": operation_open,
        "SEGURO_APAGAR_PC": safe_off,
        "ULTIMA_ACTUALIZACION_ARG": arg_time,
        "ULTIMA_ACTUALIZACION_NY": ny_time,
    }


def render_panel(status: dict[str, str] | None = None) -> str:
    status = build_status() if status is None else status
    lines = [
        "=" * 60,
        "MANIPULANTE - PANEL DE ESTADO",
        "Actualiza cada 30 segundos",
        "=" * 60,
        "",
        f"ESTADO GENERAL: {status['ESTADO_GENERAL']}",
        "",
        f"CUENTA: {status['CUENTA']}",
        f"RUNNER: {status['RUNNER']}",
        f"PID RUNNER: {status['PID_RUNNER']}",
        f"MT5: {status['MT5']}",
        "",
        f"NEWS: {status['NEWS']}",
        f"ULTIMA DECISION: {status['ULTIMA_DECISION']}",
        f"OPERACION ABIERTA: {status['OPERACION_ABIERTA']}",
        f"SEGURO APAGAR PC: {status['SEGURO_APAGAR_PC']}",
        "",
        (
            "ULTIMA ACTUALIZACION: "
            f"{status['ULTIMA_ACTUALIZACION_ARG']} ARG / "
            f"{status['ULTIMA_ACTUALIZACION_NY']} NY"
        ),
        "",
        "=" * 60,
        "SIGNIFICADO",
        "VERDE    = Todo bien",
        "AMARILLO = Bot activo pero no opera por regla",
        "ROJO     = Bot apagado o error",
        "CRITICO  = No apagar PC",
        "VIOLETA  = Revisar duplicados",
        "=" * 60,
        "",
        "CTRL+C para cerrar este panel. El bot NO se apaga.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="MANIPULANTE quick status panel")
    parser.add_argument("--runner-count", action="store_true")
    parser.add_argument("--runner-pids", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runners = find_runner_processes()
    if args.runner_count:
        print(len(runners))
        return 0
    if args.runner_pids:
        print(",".join(str(proc.pid) for proc in runners))
        return 0

    status = build_status(runners=runners)
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=True))
    else:
        print(render_panel(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
