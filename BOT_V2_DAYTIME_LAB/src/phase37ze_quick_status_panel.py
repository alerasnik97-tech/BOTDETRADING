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
STOP_BOT = ROOT / "MANIPULANTE" / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"

RUNNER_SCRIPT = "phase37_ftmo_trial_bot_runner.py"
STATUS_SCRIPT = "phase37ze_quick_status_panel.py"
STATUS_OK = "OK - BOT LISTO"
STATUS_STOPPED = "BOT DETENIDO"
STATUS_BLOCKED_HOURS = "BLOQUEADO - FUERA DE HORARIO"
STATUS_BLOCKED_NEWS = "BLOQUEADO - NOTICIAS"
STATUS_BLOCKED_SIGNAL = "BLOQUEADO - SIN SENAL"
STATUS_BLOCKED_AUTOTRADING = "BLOQUEADO - AUTOTRADING"
STATUS_ERROR = "ERROR - REVISAR SISTEMA"
STATUS_DANGER = "PELIGRO - NO APAGAR PC"
STATUS_DUPLICATE = "DUPLICADO - LIMPIAR RUNNERS"
STATUS_STALE_LOCK = "LOCK VIEJO - START LO PUEDE REPARAR"

VALID_STATES = {
    STATUS_OK,
    STATUS_STOPPED,
    STATUS_BLOCKED_HOURS,
    STATUS_BLOCKED_NEWS,
    STATUS_BLOCKED_SIGNAL,
    STATUS_BLOCKED_AUTOTRADING,
    STATUS_ERROR,
    STATUS_DANGER,
    STATUS_DUPLICATE,
    STATUS_STALE_LOCK,
}
LEGACY_STATE_MAP = {
    "VERDE": STATUS_OK,
    "AMARILLO": STATUS_BLOCKED_SIGNAL,
    "BLOQUEADO - AUTOTRADING DESHABILITADO": STATUS_BLOCKED_AUTOTRADING,
    "ROJO": STATUS_ERROR,
    "CRITICO": STATUS_DANGER,
    "VIOLETA": STATUS_DUPLICATE,
}


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
    if runner_count == 0 and operation_open == "NO":
        return "SI"
    from_qs = _yes_no(qs.get("SEGURO_APAGAR_PC") or qs.get("SAFE_TO_TURN_OFF_PC"))
    if from_qs:
        return from_qs
    if operation_open in {"SI", "DESCONOCIDO"}:
        return "NO"
    from_hb = _yes_no(hb.get("safe_to_turn_off_pc"))
    if from_hb:
        return from_hb
    # Si no hay operacion abierta, es seguro apagar (el usuario decide si cierra el bot)
    return "SI"


def _live_position_open(mt5_running: bool = True) -> str | None:
    if not mt5_running:
        return None
    try:
        from phase37_ftmo_trial_support import account_gate, open_position_status

        account = account_gate(passive=True)
        account_text = " ".join(
            str(account.get(key) or "") for key in ("company", "server", "name", "trade_mode_label", "state")
        ).lower()
        if (
            not account.get("ftmo_demo_trial_confirmed")
            or "exness" in account_text
            or str(account.get("trade_mode_label") or "").upper() == "REAL"
        ):
            return None
        status = open_position_status(passive=True)
    except Exception:
        return None
    value = status.get("position_open")
    if value is True:
        return "SI"
    if value is False:
        return "NO"
    return None


def _open_position_status(mt5_status: str, live_position: str | None, local_operation_open: str) -> str:
    if live_position == "SI":
        return "OPEN_POSITION_CONFIRMED"
    if live_position == "NO":
        return "NO_OPEN_POSITION_CONFIRMED"
    if local_operation_open == "SI":
        return "OPEN_POSITION_UNKNOWN"
    if mt5_status in {"CERRADO", "DESCONOCIDO"}:
        return "NO_OPEN_POSITION_CONFIRMED"
    return "OPEN_POSITION_UNKNOWN"


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
    stop_bot_active = STOP_BOT.exists()
    pids = ", ".join(str(proc.pid) for proc in runners) if runners else "---"
    runner_state = "ACTIVO" if runner_count > 0 else "APAGADO"
    news = str(qs.get("NEWS") or hb.get("news_gate") or "---").strip() or "---"
    decision = str(
        qs.get("ULTIMA_DECISION")
        or hb.get("last_decision")
        or last_decision_row.get("final_decision")
        or "---"
    ).strip()
    local_operation_open = _operation_open(qs, hb)
    mt5_running = (mt5 == "ABIERTO")
    live_position = _live_position_open(mt5_running=mt5_running)
    open_position_status = _open_position_status(mt5, live_position, local_operation_open)
    if open_position_status == "OPEN_POSITION_CONFIRMED":
        operation_open = "SI"
    elif open_position_status == "OPEN_POSITION_UNKNOWN":
        operation_open = "DESCONOCIDO"
    else:
        operation_open = "NO"
    safe_off = _safe_to_turn_off(qs, hb, operation_open, runner_count)
    motivo = str(qs.get("MOTIVO") or "").strip().upper()
    orders_message = str(qs.get("ORDENES") or hb.get("orders_message") or "---").strip() or "---"
    order_check = str(qs.get("ORDER_CHECK") or ("PASS" if hb.get("order_check_pass") else "---")).strip() or "---"
    order_send = str(qs.get("ORDER_SEND") or "GATEADO").strip() or "GATEADO"
    account_trade_allowed = _yes_no(qs.get("ACCOUNT_TRADE_ALLOWED") or hb.get("account_trade_allowed"))
    terminal_trade_allowed = _yes_no(qs.get("TERMINAL_TRADE_ALLOWED") or hb.get("terminal_trade_allowed"))
    tradeapi_disabled = _yes_no(qs.get("TRADEAPI_DISABLED") or hb.get("tradeapi_disabled"))
    permission_conclusion = str(
        qs.get("PERMISSION_CONCLUSION") or hb.get("permission_conclusion") or "---"
    ).strip() or "---"
    action = str(qs.get("ACCION") or hb.get("action_required") or "---").strip() or "---"

    raw_estado = str(qs.get("ESTADO_GENERAL") or "").strip().upper()
    estado = LEGACY_STATE_MAP.get(raw_estado, raw_estado)
    if estado not in VALID_STATES:
        estado = STATUS_OK
    if runner_count > 1:
        estado = STATUS_DUPLICATE
    elif operation_open == "SI" or _yes_no(hb.get("critical_position_still_open")) == "SI":
        estado = STATUS_DANGER
    elif runner_count == 0 and stop_bot_active:
        estado = STATUS_STOPPED
        action = "Use START_MANIPULANTE.bat para iniciar"
    elif runner_count == 0:
        estado = STATUS_ERROR
    elif motivo == "AUTOTRADING_DESHABILITADO" or hb.get("order_readiness_state") == "BLOCKED_AUTOTRADING_DISABLED":
        estado = STATUS_BLOCKED_AUTOTRADING
    elif news not in {"ALLOW", "---"}:
        estado = STATUS_BLOCKED_NEWS
    elif "CUTOFF" in decision or "WINDOW_CLOSED" in decision:
        estado = STATUS_BLOCKED_HOURS
    elif "WAIT_SIGNAL" in decision:
        estado = STATUS_BLOCKED_SIGNAL
    else:
        # Por defecto si no hay señal y está todo bien
        if decision == "---" and runner_count > 0:
            estado = STATUS_OK
        else:
            estado = STATUS_BLOCKED_SIGNAL

    # Check for stale lock if bot is off
    if runner_count == 0:
        # If lock exists but no runner, it's stale
        lock_file = LOG_DIR / "runner.lock"
        if lock_file.exists():
            estado = STATUS_STALE_LOCK
            action = "Use STOP y luego START para limpiar"

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
        "BOT": runner_state,
        "RUNNER": runner_state,
        "PID_RUNNER": pids,
        "MT5": mt5,
        "ORDENES": orders_message,
        "ORDER_CHECK": order_check,
        "ORDER_SEND": order_send,
        "ACCOUNT_TRADE_ALLOWED": account_trade_allowed,
        "TERMINAL_TRADE_ALLOWED": terminal_trade_allowed,
        "TRADEAPI_DISABLED": tradeapi_disabled,
        "PERMISSION_CONCLUSION": permission_conclusion,
        "ACCION": action,
        "STOP_BOT_ACTIVO": "SI" if stop_bot_active else "NO",
        "MOTIVO": "STOP_BOT ACTIVO" if stop_bot_active else (motivo or "---"),
        "NEWS": news,
        "ULTIMA_DECISION": decision or "---",
        "OPERACION_ABIERTA": operation_open,
        "OPEN_POSITION_STATUS": open_position_status,
        "SEGURO_APAGAR_PC": safe_off,
        "ULTIMA_ACTUALIZACION_ARG": arg_time,
        "ULTIMA_ACTUALIZACION_NY": ny_time,
    }


def render_panel(status: dict[str, str], mode: str = "clean") -> str:
    if mode == "technical":
        return render_panel_technical(status)
    return render_panel_clean(status)


def render_panel_clean(status: dict[str, str]) -> str:
    if status["ESTADO_GENERAL"] == STATUS_STOPPED:
        lines = [
            "=" * 60,
            "MANIPULANTE - ESTADO DEL BOT",
            "Actualiza cada 30 segundos",
            "=" * 60,
            "",
            "ESTADO: BOT DETENIDO",
            "BOT: APAGADO",
            "MOTIVO: STOP_BOT ACTIVO",
            "ACCION: Use START_MANIPULANTE.bat para iniciar",
            "",
            f"CUENTA: {status['CUENTA'].split(' / ')[0]}",
            f"OPERACION ABIERTA: {status['OPERACION_ABIERTA']}",
            f"SEGURO APAGAR PC: {status['SEGURO_APAGAR_PC']}",
            "",
            "=" * 60,
            "CTRL+C para cerrar este panel. El bot NO se apaga.",
            "=" * 60,
        ]
        return "\n".join(lines)

    # Mapeo de noticias
    noticias = "BLOQUEADO" if status["NEWS"] not in {"ALLOW", "---"} else "PERMITIDO"
    # Mapeo de ordenes
    ordenes = "LISTAS"
    if status["ESTADO_GENERAL"] in {STATUS_STOPPED, STATUS_BLOCKED_AUTOTRADING, STATUS_ERROR, STATUS_DUPLICATE}:
        ordenes = "BLOQUEADAS"
    elif status["ESTADO_GENERAL"] in {STATUS_BLOCKED_HOURS, STATUS_BLOCKED_NEWS}:
        ordenes = "BLOQUEADAS"
    # Si es STATUS_OK o STATUS_BLOCKED_SIGNAL, queda como LISTAS

    # Determinar MODO (DEMO/REAL)
    cuenta_text = status["CUENTA"].upper()
    modo_text = "DEMO"
    if "REAL" in cuenta_text or "LIVE" in cuenta_text:
        modo_text = "REAL"

    lines = [
        "=" * 60,
        "MANIPULANTE - ESTADO DEL BOT",
        "Actualiza cada 30 segundos",
        "=" * 60,
        "",
        f"ESTADO: {status['ESTADO_GENERAL']}",
        "",
        f"BOT: {status['BOT']}",
        f"CUENTA: {status['CUENTA'].split(' / ')[0]}",
        f"MODO: {modo_text}",
        f"ORDENES: {ordenes}",
        f"NOTICIAS: {noticias}",
        f"ULTIMA DECISION: {status['ULTIMA_DECISION']}",
        f"OPERACION ABIERTA: {status['OPERACION_ABIERTA']}",
        f"SEGURO APAGAR PC: {status['SEGURO_APAGAR_PC']}",
        "",
        f"HORA: {status['ULTIMA_ACTUALIZACION_ARG']} ARG / {status['ULTIMA_ACTUALIZACION_NY']} NY",
        "",
        "=" * 60,
        "QUE SIGNIFICA",
        "OK        = Bot listo para operar si aparece senal",
        "BLOQUEADO = Bot activo pero no opera por regla",
        "PELIGRO   = No apagar PC",
        "ERROR     = Revisar sistema",
        "DETENIDO  = STOP_BOT activo; START lo reactiva si es seguro",
        "DUPLICADO = Limpiar runners",
        "LOCK VIEJO = El bot se cerro mal; START lo puede reparar",
        "=" * 60,
        "",
        "CTRL+C para cerrar este panel. El bot NO se apaga.",
        "=" * 60,
    ]
    return "\n".join(lines)


def render_panel_technical(status: dict[str, str]) -> str:
    lines = [
        "=" * 60,
        "MANIPULANTE - PANEL TECNICO",
        "Actualiza cada 30 segundos",
        "=" * 60,
        "",
        f"ESTADO GENERAL: {status['ESTADO_GENERAL']}",
        "",
        f"BOT: {status['BOT']}",
        f"CUENTA: {status['CUENTA']}",
        f"RUNNER: {status['RUNNER']}",
        f"PID RUNNER: {status['PID_RUNNER']}",
        f"MT5: {status['MT5']}",
        f"ORDENES: {status['ORDENES'].replace('ORDENES: ', '')}",
        f"ORDER_CHECK: {status['ORDER_CHECK']}",
        f"ORDER_SEND: {status['ORDER_SEND']}",
        f"CUENTA TRADE: {status['ACCOUNT_TRADE_ALLOWED']}",
        f"TERMINAL TRADE: {status['TERMINAL_TRADE_ALLOWED']}",
        f"PYTHON API BLOQUEADA: {status['TRADEAPI_DISABLED']}",
        f"CONCLUSION: {status['PERMISSION_CONCLUSION']}",
        f"ACCION: {status['ACCION']}",
        f"STOP_BOT ACTIVO: {status['STOP_BOT_ACTIVO']}",
        f"MOTIVO: {status['MOTIVO']}",
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
        "CTRL+C para cerrar este panel.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="MANIPULANTE quick status panel")
    parser.add_argument("--runner-count", action="store_true")
    parser.add_argument("--runner-pids", action="store_true")
    parser.add_argument("--open-position-status", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=["clean", "technical"], default="clean")
    args = parser.parse_args()

    runners = find_runner_processes()
    if args.runner_count:
        print(len(runners))
        return 0
    if args.runner_pids:
        print(",".join(str(proc.pid) for proc in runners))
        return 0

    status = build_status(runners=runners)
    if args.open_position_status:
        print(status["OPEN_POSITION_STATUS"])
        return 0
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=True))
    else:
        print(render_panel(status, mode=args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
