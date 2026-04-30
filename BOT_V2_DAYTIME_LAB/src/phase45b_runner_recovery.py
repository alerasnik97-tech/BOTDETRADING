# -*- coding: utf-8 -*-
"""
Phase 45B - Runner Recovery Script
Handles robust detection, stale lock cleanup, and safe process termination for MANIPULANTE.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Root paths
ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MANIPULANTE = ROOT / "MANIPULANTE"
LOG_DIR = MANIPULANTE / "10_LOGS_PAPER" / "ftmo_trial_bot"
LOCK_FILE = LOG_DIR / "runner.lock"
HEARTBEAT_FILE = LOG_DIR / "heartbeat.json"
QUICK_STATUS_FILE = LOG_DIR / "quick_status.txt"
STOP_FILE = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"

RUNNER_SCRIPT = "phase37_ftmo_trial_bot_runner.py"

def get_runners():
    """Find active python processes running the runner script."""
    command = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python' -and $_.CommandLine -match '" + RUNNER_SCRIPT + "' } | "
        "Select-Object ProcessId, Name, CommandLine, CreationDate | "
        "ConvertTo-Json"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]
        
        runners = []
        current_pid = os.getpid()
        for p in data:
            pid = p.get("ProcessId")
            cmd = p.get("CommandLine") or ""
            # Exclude self and common non-runner scripts
            if pid == current_pid: continue
            if "status_panel" in cmd.lower() or "status_manipulante" in cmd.lower(): continue
            if "recovery" in cmd.lower(): continue
            
            runners.append({
                "pid": pid,
                "name": p.get("Name"),
                "command_line": cmd,
                "creation_date": p.get("CreationDate")
            })
        return runners
    except Exception as e:
        print(f"Error finding runners: {e}")
        return []

def get_lock_info():
    """Read PID from runner.lock."""
    if not LOCK_FILE.exists():
        return None
    try:
        content = LOCK_FILE.read_text().strip()
        return int(content)
    except:
        return -1

def is_position_open():
    """Check if a position is open via quick_status.txt."""
    if not QUICK_STATUS_FILE.exists():
        return None
    try:
        content = QUICK_STATUS_FILE.read_text()
        for line in content.splitlines():
            if "OPERACION_ABIERTA=SI" in line:
                return True
            if "OPERACION_ABIERTA=NO" in line:
                return False
    except:
        pass
    return None

def check_heartbeat():
    """Check if heartbeat is recent."""
    if not HEARTBEAT_FILE.exists():
        return None, None
    try:
        data = json.loads(HEARTBEAT_FILE.read_text())
        ts_str = data.get("timestamp_local")
        if not ts_str: return None, None
        ts = datetime.fromisoformat(ts_str)
        age = (datetime.now(ts.tzinfo) - ts).total_seconds()
        return ts, age
    except:
        return None, None

def diagnose():
    """Perform full diagnosis."""
    runners = get_runners()
    lock_pid = get_lock_info()
    pos_open = is_position_open()
    hb_ts, hb_age = check_heartbeat()
    
    report = {
        "runners": runners,
        "lock_pid": lock_pid,
        "position_open": pos_open,
        "heartbeat_age": hb_age,
        "status": "UNKNOWN"
    }
    
    runner_pids = [r["pid"] for r in runners]
    
    if lock_pid is None:
        if not runners:
            report["status"] = "BOT_STOPPED"
        else:
            report["status"] = "ORPHAN_RUNNER_NO_LOCK"
    elif lock_pid == -1:
        report["status"] = "LOCK_CORRUPT"
    elif lock_pid in runner_pids:
        if hb_age is not None and hb_age < 300:
            report["status"] = "LOCK_VALID"
        else:
            report["status"] = "LOCK_STUCK_OR_HEARTBEAT_DEAD"
    else:
        # Lock PID doesn't match any running runner process
        report["status"] = "LOCK_STALE"
    
    if len(runners) > 1:
        report["status"] = "DUPLICATE_RUNNERS"
        
    return report

def print_status(report):
    print("=" * 60)
    print("DIAGNOSTICO DE RUNNER MANIPULANTE")
    print("=" * 60)
    print(f"Estado Detectado: {report['status']}")
    print(f"Lock PID: {report['lock_pid']}")
    print(f"Runners Activos: {len(report['runners'])}")
    for r in report['runners']:
        print(f"  - PID: {r['pid']} | Creado: {r['creation_date']}")
    print(f"Posicion Abierta: {report['position_open']}")
    if report['heartbeat_age'] is not None:
        print(f"Edad Heartbeat: {report['heartbeat_age']:.1f}s")
    else:
        print("Heartbeat: No encontrado")
    print("=" * 60)

def clean_stale_lock(report):
    if report["status"] not in ["LOCK_STALE", "LOCK_CORRUPT"]:
        print(f"No se limpia lock: Estado {report['status']} no es STALE.")
        return False
    
    if report["position_open"] is True:
        print("ABORT: Posicion abierta detectada. No se limpia lock manualmente.")
        return False
        
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            print("runner.lock eliminado exitosamente.")
            return True
    except Exception as e:
        print(f"Error al eliminar lock: {e}")
    return False

def stop_runner_safe(report):
    if report["position_open"] is True:
        print("ABORT: Posicion abierta detectada. No se matan procesos.")
        return False
        
    for r in report["runners"]:
        pid = r["pid"]
        print(f"Cerrando runner PID {pid}...")
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
            print(f"PID {pid} terminado.")
        except Exception as e:
            print(f"Error al matar PID {pid}: {e}")
            
    if LOCK_FILE.exists() and not get_runners():
        try:
            LOCK_FILE.unlink()
            print("runner.lock limpiado tras detener runners.")
        except:
            pass
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--clean-stale-lock", action="store_true")
    parser.add_argument("--stop-runner-safe", action="store_true")
    args = parser.parse_args()
    
    rep = diagnose()
    
    if args.status:
        print_status(rep)
        
    if args.clean_stale_lock:
        clean_stale_lock(rep)
        
    if args.stop_runner_safe:
        stop_runner_safe(rep)
    
    if not any([args.status, args.clean_stale_lock, args.stop_runner_safe]):
        print_status(rep)
