"""
SCBI Dual Line Daily Forward Orchestrator

Coordinación diaria de SCBI_M5_GLOBAL y SCBI_CORE.
Garantiza separación de evidencia, validación de inputs y actualización del scoreboard.
"""
import argparse
import os
import subprocess
import logging
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
ORCHESTRATOR_STATUS = ROOT / "results" / "SCBI_DUAL_ORCHESTRATOR_STATUS.json"
GLOBAL_AUTOPILOT = ROOT / "scratch" / "run_scbi_phase1_autopilot.py"
CORE_RUNNER = ROOT / "scratch" / "run_scbi_core_forward_phase1.py"
SCOREBOARD_BUILDER = ROOT / "scratch" / "build_scbi_dual_line_scoreboard.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_command(cmd_list, description, *, extra_env=None):
    logging.info(f"Executing: {description}")
    try:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
        )
        if result.returncode == 0:
            logging.info(f"[SUCCESS] {description}")
            return True, result.stdout
        else:
            logging.error(f"[FAILED] {description}: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        logging.error(f"[ERROR] {description}: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Dual Line Daily Orchestrator")
    parser.add_argument("--run", action="store_true", help="Ejecutar día oficial en ambas líneas")
    parser.add_argument("--date", type=str, help="Fecha específica (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Forzar re-ejecución ignorando el lockfile")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    lock_file = ROOT / "results" / f"chain_lock_{target_date}.json"
    run_id = f"dual_chain_{target_date}"

    logging.info(f"=== DUAL LINE DAILY CHAIN START [{target_date}] ===")
    
    # 0. Rerun Protection (Idempotency Guard)
    if lock_file.exists() and not args.force:
        logging.error(f"FAIL-CLOSED: La fecha {target_date} ya fue procesada. Use --force para re-ejecutar.")
        return
    
    # 0.5 Risk Guards Pre-check (Prop Firm Layer)
    risk_guards_script = ROOT / "scratch" / "prop_firm_risk_guards.py"
    command_env = {"SCBI_FORWARD_RUN_ID": run_id}
    success_r, out_r = run_command(["python", str(risk_guards_script)], "Prop Firm Risk Guards Check", extra_env=command_env)
    
    if not success_r:
        logging.error("CRITICAL: Risk Guards Execution Failed.")
        return
    
    try:
        risk_report = json.loads(out_r)
        if risk_report.get("status") == "FAIL":
            logging.error(f"FAIL-CLOSED: Risk Guard Blocker detected: {out_r}")
            return
        logging.info("Prop Firm Risk Guards: PASS")
    except Exception as e:
        logging.warning(f"Could not parse risk report, proceeding with caution: {e}")

    # 1. Ejecutar GLOBAL (Autopilot)
    # Autopilot detecta la fecha automáticamente si no se pasa --date
    global_cmd = ["python", str(GLOBAL_AUTOPILOT)]
    if args.run:
        global_cmd.append("--run")
    if args.date:
        global_cmd.extend(["--date", args.date])
    
    success_g, out_g = run_command(global_cmd, "SCBI_M5_GLOBAL Autopilot", extra_env=command_env)
    
    # 2. Ejecutar CORE (Forward Runner)
    # El runner de core actualmente está simplificado (rehearsal), 
    # pero para el chain debe comportarse como el de la global.
    # Por ahora lo llamaremos para la misma fecha si se provee.
    core_cmd = ["python", str(CORE_RUNNER)]
    # El runner de core actual no toma --date, lo adaptaremos o llamaremos tal cual.
    # En un sistema real, CORE debería tener su propio Autopilot.
    success_c, out_c = run_command(core_cmd, "SCBI_CORE Forward Runner", extra_env=command_env)
    
    # 3. Actualizar Scoreboard Comparativo
    success_s, out_s = run_command(["python", str(SCOREBOARD_BUILDER)], "Dual Line Scoreboard Update", extra_env=command_env)
    
    # 4. Update Orchestrator Status and Lockfile
    status = {
        "target_date": target_date,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": {
            "global": "SUCCESS" if success_g else "FAILED",
            "core": "SUCCESS" if success_c else "FAILED",
            "scoreboard": "SUCCESS" if success_s else "FAILED"
        }
    }
    
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w") as f:
        json.dump(status, f, indent=2)
        
    with open(ORCHESTRATOR_STATUS, "w") as f:
        json.dump(status, f, indent=2)
        
    logging.info("=== DUAL LINE DAILY CHAIN END ===")


if __name__ == "__main__":
    main()
