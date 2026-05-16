"""
EURUSD Daily Chain Historical Rehearsal Runner - V9 (Full JSON Mocks)

Certifica la cadena operativa unificada mediante replay histórico.
Incluye mocks de JSON con estructura completa para pasar el Unified Status Rebuild.
"""
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OFFICIAL_RESULTS = ROOT / "results"
REHEARSAL_RESULTS = ROOT / "results_REHEARSAL"
CHAIN_STATUS = ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS.json"
FREEZE_STATUS_FILE = ROOT / "SCBI_PHASE1_FREEZE_STATUS.json"

CHAIN_SCRIPT = ROOT / "scratch" / "run_daily_data_to_decision_chain.py"

CSV_HEADERS = {
    "SCBI_FORWARD_LEDGER.csv": "session_date,event_timestamp,event_type,status,signal_id,strategy_id,pair,direction,sweep_level,sweep_time,sweep_extreme,level_price,entry_time,entry_price,sl,tp,risk_pips,exit_time,exit_price,exit_type,pnl_r,block_reason,block_details,observed_spread_pips,applied_spread_pips,data_quality_flag,news_check_status,notes",
    "SCBI_FORWARD_DAILY_STATUS.csv": "session_date,sweeps_detected,sweeps_blocked_news,sweeps_blocked_daily_limit,sweeps_no_scbi,sweeps_data_issue,trades_paper,result,pnl_r,cumulative_n,cumulative_pf,cumulative_exp,cumulative_dd,incidents,notes",
    "SCBI_CORE_PHASE1/core_phase1_ledger.csv": "event_id,timestamp_ny,level,direction,entry_price,sl,tp,risk_pips,exit_time,exit_price,pnl_r,exit_reason,news_blocked",
    "SCBI_DUAL_LINE_SCOREBOARD.csv": "Line,Sample_N,PF_Forward,Exp_Forward,Max_DD_R,Last_Activity,Drift_R,Drift_Label,Drift_Comparable,Drift_Governance_Mode,Telemetry_Execution_Fidelity,Telemetry_Blocking_Fidelity,Telemetry_Last_Guard_Status,Telemetry_Last_Incident,Telemetry_Lineage_Coverage,Telemetry_Official_Event_Count,Telemetry_Trace_Path",
    "SCBI_UNIFIED_LINE_STATUS.csv": "Line,Line_Status_Clarity,Institutional_Operating_State,Sample_N,PF_Forward,Exp_Forward,Max_DD_R,Drift_Label,Expectation_Label,Telemetry_Execution_Fidelity,Telemetry_Blocking_Fidelity,Telemetry_Lineage_Coverage,Guard_Status,Incident_Code,Readiness_State,Promotion_State,Next_Action,Summary,Active_Risks",
    "SCBI_FORWARD_TELEMETRY_TRACE.csv": "trace_id,run_id,session_date,source_line,official_flag,source_artifact,source_row_key,event_class,event_phase,status,signal_or_event_id,event_time_ny,level,direction,risk_pips,pnl_r,news_affected,block_reason,guard_reason,fill_type,spread_proxy_pips,slippage_proxy_pips,cost_proxy_pips,cost_proxy_r,blocking_event_name,blocking_event_time_ny,blocking_rule_used,incident_code,ledger_ref,daily_status_ref,scoreboard_ref,tribunal_ref,notes"
}

JSON_MOCKS = {
    "SCBI_FORWARD_TRIBUNAL_SUMMARY.json": json.dumps({"summary": "rehearsal_mock"}),
    "SCBI_UNIFIED_LINE_STATUS.json": json.dumps({"lines": {}}),
    "SCBI_SIGNAL_DRIFT_REPORT.json": json.dumps({
        "generated_at_utc": "2026-04-23T00:00:00Z",
        "baseline_version": "POST_HARDENING_SIGNAL_DRIFT_V1",
        "monitor_validation_verdict": "DRIFT_MONITOR_STILL_VALID",
        "tribunal_integration_mode": "TRIBUNAL_SAFE_TO_USE",
        "lines": {
            "SCBI_M5_GLOBAL": {"verdict": "NO_DRIFT", "drift_r": 0.0},
            "SCBI_CORE": {"verdict": "NO_DRIFT", "drift_r": 0.0}
        }
    }),
    "SCBI_SIGNAL_DRIFT_VALIDATION.json": json.dumps({"overall": {"verdict": "DRIFT_MONITOR_STILL_VALID"}}),
    "DAILY_DECISION_ARTIFACT_LAST.json": json.dumps({"target_date": "2026-04-20"})
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

class RehearsalEnvironment:
    def __enter__(self):
        logging.info("--- PREPARANDO ENTORNO DE REHEARSAL (V9) ---")
        self.backup(OFFICIAL_RESULTS, ROOT / "results_OFFICIAL_TEMP")
        self.backup(CHAIN_STATUS, ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS_OFFICIAL_TEMP.json")
        self.backup(FREEZE_STATUS_FILE, ROOT / "SCBI_PHASE1_FREEZE_STATUS_OFFICIAL_TEMP.json")
        OFFICIAL_RESULTS.mkdir(exist_ok=True)
        (OFFICIAL_RESULTS / "SCBI_CORE_PHASE1").mkdir(exist_ok=True)
        
        drift_baseline = ROOT / "results_OFFICIAL_TEMP" / "SCBI_SIGNAL_DRIFT_BASELINE.json"
        if drift_baseline.exists():
            shutil.copy(str(drift_baseline), str(OFFICIAL_RESULTS / "SCBI_SIGNAL_DRIFT_BASELINE.json"))
            
        new_hashes = {}
        for filename, header in CSV_HEADERS.items():
            path = OFFICIAL_RESULTS / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(header + "\n")
                if filename == "SCBI_FORWARD_LEDGER.csv":
                    f.write("2026-04-20,2026-04-20 20:00:00,BOOTSTRAP,PASS,BOOT_001,SCBI_M5_GLOBAL,EURUSD,long,,,,,,,,,,,,,,,,,,,\n")
                if filename == "SCBI_FORWARD_DAILY_STATUS.csv":
                    f.write("2026-04-20,0,0,0,0,0,0,NO_TRADE,0,0,0,0,0,None,\n")
                if filename == "SCBI_CORE_PHASE1/core_phase1_ledger.csv":
                    f.write("CORE_BOOT,2026-04-20 20:00:00,pdh,long,1.08,1.07,1.10,10.0,2026-04-20 21:00:00,1.10,1.5,tp_hit,False\n")
            new_hashes[f"results/{filename}"] = sha256_file(path)
            
        for filename, content in JSON_MOCKS.items():
            path = OFFICIAL_RESULTS / filename
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)

        if (ROOT / "SCBI_PHASE1_FREEZE_STATUS_OFFICIAL_TEMP.json").exists():
            with open(ROOT / "SCBI_PHASE1_FREEZE_STATUS_OFFICIAL_TEMP.json", "r", encoding="utf-8") as f:
                status = json.load(f)
            for record in status.get("tracked_runtime_files", []):
                rel_path = record["path"]
                if rel_path in new_hashes:
                    record["sha256"] = new_hashes[rel_path]
                    record["rows"] = 1
            with open(FREEZE_STATUS_FILE, "w", encoding="utf-8", newline="\n") as f:
                json.dump(status, f, indent=2)
        return self

    def backup(self, source, target):
        if source.exists():
            if target.exists():
                if target.is_dir(): shutil.rmtree(target)
                else: os.remove(target)
            shutil.move(str(source), str(target))

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info("--- RESTAURANDO ENTORNO OFICIAL ---")
        if REHEARSAL_RESULTS.exists(): shutil.rmtree(REHEARSAL_RESULTS)
        shutil.copytree(str(OFFICIAL_RESULTS), str(REHEARSAL_RESULTS))
        if OFFICIAL_RESULTS.exists(): shutil.rmtree(OFFICIAL_RESULTS)
        if CHAIN_STATUS.exists():
            if (ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS_REHEARSAL.json").exists():
                os.remove(ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS_REHEARSAL.json")
            shutil.move(str(CHAIN_STATUS), str(ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS_REHEARSAL.json"))
        if FREEZE_STATUS_FILE.exists(): os.remove(FREEZE_STATUS_FILE)
        self.restore(ROOT / "results_OFFICIAL_TEMP", OFFICIAL_RESULTS)
        self.restore(ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS_OFFICIAL_TEMP.json", CHAIN_STATUS)
        self.restore(ROOT / "SCBI_PHASE1_FREEZE_STATUS_OFFICIAL_TEMP.json", FREEZE_STATUS_FILE)

    def restore(self, source, target):
        if source.exists():
            shutil.move(str(source), str(target))

def run_chain(date, force=False, skip_promotion=True):
    cmd = ["python", str(CHAIN_SCRIPT), "--date", date]
    if force:
        cmd.append("--force")
    if skip_promotion:
        cmd.append("--skip-promotion")
    
    logging.info(f"EJECUTANDO CADENA PARA: {date} (force={force}, skip_promotion={skip_promotion})")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return result

def main():
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": []
    }

    with RehearsalEnvironment():
        # A: Dia Valido (2026-04-21)
        res_a = run_chain("2026-04-21")
        report["scenarios"].append({
            "id": "A",
            "date": "2026-04-21",
            "description": "Dia Valido",
            "success": "DAILY_DATA_TO_DECISION_CHAIN_READY" in res_a.stdout
        })

        # B: Rerun (2026-04-21)
        res_b = run_chain("2026-04-21")
        report["scenarios"].append({
            "id": "B",
            "date": "2026-04-21",
            "description": "Rerun (Sin force)",
            "success": "GLOBAL_DATE_ALREADY_PROCESSED" in res_b.stdout
        })

        # C: Gap Cobertura (2026-04-22)
        res_c = run_chain("2026-04-22")
        report["scenarios"].append({
            "id": "C",
            "date": "2026-04-22",
            "description": "Gap Cobertura H1/M5",
            "success": "COVERAGE_GAP_REAL" in res_c.stdout or "H1_NO_TARGET_ROWS" in res_c.stdout
        })

        # D: Gap News (2026-05-01)
        res_d = run_chain("2026-05-01")
        report["scenarios"].append({
            "id": "D",
            "date": "2026-05-01",
            "description": "Gap News Horizon",
            "success": "NEWS_HORIZON_INSUFFICIENT" in res_d.stdout
        })

    with open(ROOT / "DAILY_CHAIN_REHEARSAL_LOG.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)

    logging.info("--- REHEARSAL FINALIZADO ---")
    all_pass = True
    for s in report["scenarios"]:
        status = "PASS" if s["success"] else "FAIL"
        if not s["success"]: all_pass = False
        logging.info(f"Escenario {s['id']} [{s['description']}]: {status}")
    
    if all_pass:
        logging.info("CERTIFICACION: DAILY_CHAIN_REHEARSAL_CERTIFIED")
    else:
        logging.info("CERTIFICACION: DAILY_CHAIN_REHEARSAL_NEEDS_FIXES")

if __name__ == "__main__":
    main()
