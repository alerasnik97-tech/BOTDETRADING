"""
EURUSD Lab One-Command Autopilot (V1.0)
Automatizacion operativa de cobertura, baseline, chain y bundle.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SCRATCH_DIR = ROOT / "scratch"

# Componentes
PREFLIGHT = SCRIPTS_DIR / "preflight_project_boundary_check.py"
PROMOTION = SCRATCH_DIR / "run_data_coverage_promotion.py"
DAILY_CHAIN = SCRATCH_DIR / "run_daily_data_to_decision_chain.py"
BUNDLE_BUILDER = SCRIPTS_DIR / "build_chatgpt_bundle.py"
INTAKE_DIR = ROOT / "data" / "coverage_pipeline" / "intake"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "LAB_AUTOPILOT.log", encoding="utf-8")
    ]
)

def run_script(cmd_list, description):
    logging.info(f"--- FASE: {description} ---")
    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            encoding="utf-8"
        )
        if result.returncode == 0:
            logging.info(f"[PASS] {description}")
            return True, result.stdout
        else:
            logging.error(f"[BLOCK] {description} (Exit Code: {result.returncode})")
            return False, result.stdout
    except Exception as e:
        logging.error(f"[ERROR] {description}: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Lab One-Command Autopilot")
    parser.add_argument("--date", type=str, help="Fecha objetivo YYYY-MM-DD (default: today)")
    parser.add_argument("--force-bundle", action="store_true", help="Forzar reconstruccion del bundle")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    logging.info(f"=== INICIANDO AUTOPILOT PARA {target_date} ===")

    # 1. Preflight
    success, _ = run_script(["python", str(PREFLIGHT)], "Preflight de Perimetro")
    if not success:
        print(json.dumps({"verdict": "BLOCKED_FOR_SAFETY", "reason": "Boundary check failed"}, indent=2))
        return

    # 2. Intake Detection & Promotion
    files_in_intake = list(INTAKE_DIR.glob("*.csv"))
    data_promoted = False
    if files_in_intake:
        logging.info(f"Detectados {len(files_in_intake)} archivos en intake. Iniciando promocion...")
        promo_success, out = run_script(["python", str(PROMOTION), "--dataset", "ALL", "--target-date", target_date, "--promote"], "Promocion Automatica")
        if promo_success:
            data_promoted = True
        else:
            logging.warning("La promocion fallo o no encontro data valida para esta fecha.")
    else:
        logging.info("No hay archivos nuevos en intake. Saltando fase de promocion.")

    # 3. Daily Chain (Coverage -> Baseline -> Official Run -> Tribunal -> Status)
    chain_success, chain_out = run_script(["python", str(DAILY_CHAIN), "--date", target_date], "Daily Chain")
    
    # 4. Bundle Rebuild
    bundle_refreshed = False
    if chain_success or data_promoted or args.force_bundle:
        logging.info("Cambios detectados o solicitados. Reconstruyendo bundle de handoff...")
        bundle_success, _ = run_script(["python", str(BUNDLE_BUILDER)], "Bundle Rebuild")
        bundle_refreshed = bundle_success

    # 5. Veredicto Final (Leido del reporte de la cadena)
    final_decision = "WAITING_FOR_EXTERNAL_DATA"
    try:
        chain_report_path = ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS.json"
        if chain_report_path.exists():
            with open(chain_report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
                chain_final = report_data.get("final_verdict", "UNKNOWN")
                
                if chain_final == "SUCCESS":
                    final_decision = "DAILY_CHAIN_EXECUTED"
                elif chain_final == "BLOCKED":
                    reason = report_data.get("block_reason", "")
                    if "coverage" in reason.lower() or "no_target_rows" in chain_out.lower():
                        final_decision = "MANUAL_INTAKE_REQUIRED"
                    else:
                        final_decision = "FAIL_CLOSED_CORRECT_BEHAVIOR"
                elif chain_final == "FAILED":
                    final_decision = "FAIL_CLOSED_CORRECT_BEHAVIOR"
    except Exception as e:
        logging.error(f"Error leyendo veredicto de cadena: {e}")

    if final_decision == "WAITING_FOR_EXTERNAL_DATA" and bundle_refreshed and not data_promoted:
        final_decision = "BUNDLE_REFRESH_OK"

    report = {
        "target_date": target_date,
        "automation_verdict": final_decision,
        "data_promoted": data_promoted,
        "daily_chain_success": chain_success,
        "bundle_refreshed": bundle_refreshed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }

    print("\n============================================================")
    print("REPORTE FINAL DE AUTOPILOT")
    print("============================================================")
    print(json.dumps(report, indent=2))
    print(f"\nCONCLUSION GLOBAL: {final_decision}")

if __name__ == "__main__":
    main()
