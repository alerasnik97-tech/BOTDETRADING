"""
SCBI Phase 1 Multi-Day Autopilot
================================
Operación automática día por día con protección fail-closed.

Uso:
  python scratch/run_scbi_phase1_autopilot.py [--run] [--weekly]
  
Sin --run: modo dry-run (solo detecta y reporta, no ejecuta)
Con --run: ejecuta el siguiente día hábil si pasa validación
Con --weekly: fuerza generación de weekly review si es viernes

Protocolo:
  1. Detectar última fecha corrida (ledger)
  2. Calcular siguiente día hábil (skip fines de semana)
  3. Verificar cobertura de datos (H1, M5, news)
  4. Ejecutar precheck de baseline
  5. Si todo PASS, ejecutar día oficial
  6. Si FAIL, registrar bloqueo sin ejecutar
  7. Si es viernes/sábado, generar weekly review
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
LEDGER_CSV = ROOT / "results" / "SCBI_FORWARD_LEDGER.csv"
STATUS_CSV = ROOT / "results" / "SCBI_FORWARD_DAILY_STATUS.csv"
VALIDATOR = ROOT / "scratch" / "validate_scbi_phase1_baseline.py"
RUNNER = ROOT / "scratch" / "run_scbi_forward_phase1.py"
WEEKLY = ROOT / "scratch" / "generate_scbi_phase1_weekly_review.py"
DATA_H1 = ROOT / "data_candidates_2022_2025" / "prepared" / "EURUSD_H1.csv"
DATA_M5 = ROOT / "data_candidates_2022_2025" / "prepared" / "EURUSD_M5.csv"
DATA_NEWS = ROOT / "data" / "news_eurusd_am_fortress_v3.csv"
AUTOPILOT_STATUS = ROOT / "SCBI_PHASE1_AUTOPILOT_STATUS.json"
AUTOPILOT_LOG = ROOT / "SCBI_PHASE1_AUTOPILOT_LOG.md"


def log_event(message: str):
    """Append-only logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(AUTOPILOT_LOG, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)


def get_last_trading_day():
    """Detectar última fecha corrida desde el ledger."""
    if not LEDGER_CSV.exists():
        return None
    
    dates = set()
    with open(LEDGER_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('session_date'):
                dates.add(row['session_date'])
    
    if not dates:
        return None
    
    return max(dates)


def get_next_trading_day(last_date: str) -> str:
    """Calcular siguiente día hábil (excluye sábados y domingos)."""
    last = datetime.strptime(last_date, "%Y-%m-%d")
    next_day = last + timedelta(days=1)
    
    # Skip weekend
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    
    return next_day.strftime("%Y-%m-%d")


def check_data_coverage(target_date: str) -> tuple:
    """
    Verificar que existan datos para el día objetivo.
    Retorna (bool, str): (tiene_cobertura, mensaje)
    """
    # Check H1
    if not DATA_H1.exists():
        return False, "DATA_H1 missing"
    
    h1_has_data = False
    with open(DATA_H1, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0].startswith(target_date):
                h1_has_data = True
                break
    
    if not h1_has_data:
        return False, f"H1 feed no coverage for {target_date}"
    
    # Check M5
    if not DATA_M5.exists():
        return False, "DATA_M5 missing"
    
    m5_has_data = False
    with open(DATA_M5, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0].startswith(target_date):
                m5_has_data = True
                break
    
    if not m5_has_data:
        return False, f"M5 feed no coverage for {target_date}"
    
    return True, "Data coverage OK"


def run_precheck(date: str) -> tuple:
    """
    Ejecutar validación de baseline.
    Retorna (bool, str): (pass, output)
    """
    try:
        result = subprocess.run(
            ["python", str(VALIDATOR), "--check", "--date", date],
            capture_output=True,
            text=True,
            cwd=str(ROOT)
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0 and "[PASS]" in output
        return passed, output
    except Exception as e:
        return False, f"Validator error: {e}"


def run_official_day(date: str) -> tuple:
    """
    Ejecutar día oficial.
    Retorna (bool, str): (success, output)
    """
    try:
        result = subprocess.run(
            ["python", str(RUNNER), "--date", date],
            capture_output=True,
            text=True,
            cwd=str(ROOT)
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0 and "[SUCCESS]" in output
        return success, output
    except Exception as e:
        return False, f"Runner error: {e}"


def is_weekly_review_day() -> bool:
    """Determinar si hoy es día de weekly review (viernes o sábado)."""
    today = datetime.now().weekday()
    return today in [4, 5]  # Friday = 4, Saturday = 5


def run_weekly_review() -> tuple:
    """
    Ejecutar weekly review.
    Retorna (bool, str): (success, output)
    """
    try:
        result = subprocess.run(
            ["python", str(WEEKLY)],
            capture_output=True,
            text=True,
            cwd=str(ROOT)
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output
    except Exception as e:
        return False, f"Weekly review error: {e}"


def update_status(status_dict: dict):
    """Update autopilot status file."""
    with open(AUTOPILOT_STATUS, 'w') as f:
        json.dump(status_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SCBI Phase 1 Autopilot")
    parser.add_argument("--run", action="store_true", help="Ejecutar día oficial si pasa validación")
    parser.add_argument("--weekly", action="store_true", help="Forzar weekly review")
    parser.add_argument("--date", type=str, help="Fecha específica (override automático)")
    args = parser.parse_args()
    
    log_event("=== SCBI Phase 1 Autopilot Start ===")
    
    # 1. Detectar última fecha
    last_date = get_last_trading_day()
    if not last_date:
        log_event("[ERROR] No se pudo detectar última fecha corrida (ledger vacío o missing)")
        update_status({
            "status": "BLOCKED",
            "reason": "LEDGER_MISSING_OR_EMPTY",
            "timestamp": datetime.now().isoformat()
        })
        sys.exit(1)
    
    log_event(f"[INFO] Última fecha corrida: {last_date}")
    
    # 2. Determinar siguiente fecha
    if args.date:
        next_date = args.date
        log_event(f"[INFO] Fecha override: {next_date}")
    else:
        next_date = get_next_trading_day(last_date)
        log_event(f"[INFO] Siguiente día hábil calculado: {next_date}")
    
    # 3. Verificar cobertura de datos
    has_coverage, coverage_msg = check_data_coverage(next_date)
    if not has_coverage:
        log_event(f"[BLOCKED] {coverage_msg}")
        update_status({
            "status": "BLOCKED_FOR_DATA_GAP",
            "last_date": last_date,
            "next_date": next_date,
            "reason": coverage_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        # Weekly review aunque haya data gap
        if args.weekly or is_weekly_review_day():
            log_event("[WEEKLY] Ejecutando weekly review...")
            success, output = run_weekly_review()
            if success:
                log_event("[WEEKLY] Weekly review completado")
            else:
                log_event(f"[WEEKLY] Weekly review falló: {output}")
        
        sys.exit(0)  # Exit gracefully, esperando datos
    
    log_event(f"[PASS] {coverage_msg}")
    
    # 4. Precheck de baseline
    precheck_pass, precheck_output = run_precheck(next_date)
    if not precheck_pass:
        log_event(f"[BLOCKED] Precheck falló: {precheck_output}")
        update_status({
            "status": "BLOCKED_FOR_BASELINE_DRIFT",
            "last_date": last_date,
            "next_date": next_date,
            "precheck_output": precheck_output,
            "timestamp": datetime.now().isoformat()
        })
        sys.exit(1)
    
    log_event("[PASS] Precheck baseline superado")
    
    # 5. Verificar que no sea re-run (fecha ya en ledger)
    with open(LEDGER_CSV, 'r') as f:
        content = f.read()
        if f"{next_date}," in content or f"{next_date}\n" in content:
            log_event(f"[BLOCKED] Fecha {next_date} ya existe en ledger (re-run protection)")
            update_status({
                "status": "BLOCKED_FOR_RERUN",
                "last_date": last_date,
                "next_date": next_date,
                "reason": "Date already in ledger",
                "timestamp": datetime.now().isoformat()
            })
            sys.exit(1)
    
    # 6. Ejecutar o simular
    if args.run:
        log_event(f"[EXECUTE] Corriendo día oficial: {next_date}")
        success, output = run_official_day(next_date)
        
        if success:
            log_event(f"[SUCCESS] Día {next_date} completado exitosamente")
            log_event(f"[OUTPUT] {output}")
            update_status({
                "status": "DAY_EXECUTED",
                "last_date": last_date,
                "executed_date": next_date,
                "next_date": get_next_trading_day(next_date),
                "timestamp": datetime.now().isoformat()
            })
        else:
            log_event(f"[FAILED] Ejecución falló: {output}")
            update_status({
                "status": "EXECUTION_FAILED",
                "last_date": last_date,
                "attempted_date": next_date,
                "error": output,
                "timestamp": datetime.now().isoformat()
            })
            sys.exit(1)
    else:
        log_event(f"[DRY-RUN] Todo listo para ejecutar {next_date}, use --run para operar")
        update_status({
            "status": "READY_TO_RUN",
            "last_date": last_date,
            "next_date": next_date,
            "timestamp": datetime.now().isoformat()
        })
    
    # 7. Weekly review si aplica
    if args.weekly or is_weekly_review_day():
        log_event("[WEEKLY] Ejecutando weekly review...")
        success, output = run_weekly_review()
        if success:
            log_event("[WEEKLY] Weekly review completado")
        else:
            log_event(f"[WEEKLY] Weekly review falló: {output}")
    
    log_event("=== SCBI Phase 1 Autopilot End ===")


if __name__ == "__main__":
    main()
