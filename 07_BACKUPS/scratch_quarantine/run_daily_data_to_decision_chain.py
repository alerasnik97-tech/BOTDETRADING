"""
EURUSD Daily Data to Decision Operation Chain Orchestrator

Cadena operativa única, canónica y auditable.
Desde el preflight y la cobertura de datos hasta la decisión institucional final.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")

# Rutas de scripts
PREFLIGHT = ROOT / "scripts" / "preflight_project_boundary_check.py"
COVERAGE_CHECK = ROOT / "scratch" / "run_data_coverage_check.py"
DATA_PROMOTION = ROOT / "scratch" / "run_data_coverage_promotion.py"
BASELINE_VALIDATOR = ROOT / "scratch" / "validate_scbi_phase1_baseline.py"
DUAL_CHAIN = ROOT / "scratch" / "run_dual_line_daily_chain.py"
TRIBUNAL = ROOT / "scratch" / "run_forward_evidence_tribunal.py"
UNIFIED_STATUS = ROOT / "scratch" / "run_unified_line_status_builder.py"

# Salidas
RESULTS_DIR = ROOT / "results"
DAILY_VERDICT_PATH = RESULTS_DIR / "DAILY_DECISION_ARTIFACT_LAST.json"
CHAIN_STATUS_PATH = ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "DAILY_CHAIN_EXECUTION.log", encoding="utf-8")
    ]
)

def run_step(cmd_list, description):
    logging.info(f"--- INICIANDO: {description} ---")
    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            encoding="utf-8"
        )
        if result.returncode == 0:
            logging.info(f"[SUCCESS] {description}")
            return True, result.stdout
        else:
            logging.error(f"[FAILED] {description} (Exit Code: {result.returncode})")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            return False, f"{result.stdout}\n{result.stderr}"
    except Exception as e:
        logging.error(f"[ERROR] {description}: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Daily Data to Decision Chain")
    parser.add_argument("--date", type=str, help="Fecha objetivo YYYY-MM-DD (default: today)")
    parser.add_argument("--skip-promotion", action="store_true", help="Saltar promoción de datos")
    parser.add_argument("--force", action="store_true", help="Forzar ejecución ignorando bloqueos previos")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    logging.info(f"=== DAILY OPERATION CHAIN START [{target_date}] ===")

    chain_report = {
        "target_date": target_date,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": {},
        "final_verdict": "NOT_EXECUTED",
        "block_reason": None
    }

    def update_report(step_id, success, output):
        chain_report["steps"][step_id] = {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": output[:500] + "..." if len(output) > 500 else output
        }

    # A) Project Boundary Preflight
    success, out = run_step(["python", str(PREFLIGHT)], "A) Project Boundary Preflight")
    update_report("project_boundary_preflight", success, out)
    if not success:
        chain_report["final_verdict"] = "BLOCKED"
        chain_report["block_reason"] = "Preflight boundary check failed."
        goto_end(chain_report)
        return

    # B) Data Coverage Check
    success, out = run_step(["python", str(COVERAGE_CHECK), "--target-date", target_date], "B) Data Coverage Check")
    update_report("data_coverage_check", success, out)
    if not success:
        chain_report["final_verdict"] = "BLOCKED"
        chain_report["block_reason"] = "Insufficient data coverage or validator pre-block."
        goto_end(chain_report)
        return

    # C) Data Promotion
    if not args.skip_promotion:
        success, out = run_step(["python", str(DATA_PROMOTION), "--dataset", "ALL", "--target-date", target_date, "--promote"], "C) Data Promotion")
        update_report("data_promotion", success, out)
        if not success:
            chain_report["final_verdict"] = "BLOCKED"
            chain_report["block_reason"] = "Data promotion failed."
            goto_end(chain_report)
            return
    else:
        logging.info("Skipping C) Data Promotion per request.")
        update_report("data_promotion", True, "Skipped")

    # D) Baseline Validator
    val_cmd = ["python", str(BASELINE_VALIDATOR), "--check", "--date", target_date]
    success, out = run_step(val_cmd, "D) Baseline Validator")
    update_report("baseline_validator", success, out)
    if not success:
        chain_report["final_verdict"] = "BLOCKED"
        chain_report["block_reason"] = "Baseline integrity drift detected."
        goto_end(chain_report)
        return

    # E) Official Run (Global + Core + Scoreboard)
    dual_cmd = ["python", str(DUAL_CHAIN), "--run", "--date", target_date]
    if args.force:
        dual_cmd.append("--force")
    success, out = run_step(dual_cmd, "E) Official Run & Scoreboard")
    update_report("official_run", success, out)
    if not success:
        chain_report["final_verdict"] = "FAILED"
        chain_report["block_reason"] = "Execution of trading lines or scoreboard update failed."
        goto_end(chain_report)
        return

    # G) Tribunal Rebuild
    success, out = run_step(["python", str(TRIBUNAL)], "G) Tribunal Rebuild")
    update_report("tribunal_rebuild", success, out)
    if not success:
        chain_report["final_verdict"] = "FAILED"
        chain_report["block_reason"] = "Tribunal adjudication failed."
        goto_end(chain_report)
        return

    # H) Unified Status Rebuild
    success, out = run_step(["python", str(UNIFIED_STATUS)], "H) Unified Status Rebuild")
    update_report("unified_status_rebuild", success, out)
    if not success:
        chain_report["final_verdict"] = "FAILED"
        chain_report["block_reason"] = "Unified status generation failed."
        goto_end(chain_report)
        return

    # I) Daily Decision Emit
    chain_report["final_verdict"] = "SUCCESS"
    chain_report["status_final"] = "READY_FOR_DECISION"
    goto_end(chain_report)

def goto_end(report):
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CHAIN_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    # Emit Daily Decision Artifact (Summary for the operator)
    verdict = {
        "target_date": report["target_date"],
        "valid_day": report["final_verdict"] == "SUCCESS",
        "block_reason": report["block_reason"],
        "steps_completed": list(report["steps"].keys()),
        "final_verdict": "DAILY_DATA_TO_DECISION_CHAIN_READY" if report["final_verdict"] == "SUCCESS" else "DAILY_DATA_TO_DECISION_CHAIN_NOT_READY",
        "next_action": "Review results/SCBI_UNIFIED_LINE_STATUS.json" if report["final_verdict"] == "SUCCESS" else "Check logs and fix blockers."
    }
    
    with open(DAILY_VERDICT_PATH, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    logging.info(f"=== DAILY OPERATION CHAIN END: {verdict['final_verdict']} ===")
    print(json.dumps(verdict, indent=2))

if __name__ == "__main__":
    main()
