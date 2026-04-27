"""
Forward Evidence Tribunal - Institutional Judgment Motor
Reads Dual Line Scoreboard and emits automatic verdicts based on hard thresholds.
"""
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
SCOREBOARD = ROOT / "results" / "SCBI_DUAL_LINE_SCOREBOARD.csv"
RISK_STATUS = ROOT / "results" / "SCBI_DUAL_ORCHESTRATOR_STATUS.json"
TRIBUNAL_SUMMARY = ROOT / "results" / "SCBI_FORWARD_TRIBUNAL_SUMMARY.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def judge_line(row, risk_fail=False):
    """Applies the tribunal logic to a single line row from the scoreboard."""
    n = row["Sample_N"]
    pf = row["PF_Forward"]
    dd = row["Max_DD_R"]
    
    # Pre-checks críticos
    if risk_fail:
        return "SUSPENDED (Risk Blocker)"
    if dd <= -6.0:
        return "SUSPENDED (DD Breach)"
    
    # Lógica por Checkpoints
    if n < 10:
        return "PAPER_ONLY (Gathering Sample)"
        
    if 10 <= n < 20:
        if pf < 1.0:
            return "PROMOTION_BLOCKED (Early Failure)"
        return "PAPER_ONLY (Gathering Sample)"
        
    if 20 <= n < 40:
        if pf > 2.2 and dd > -5.0:
            return "DEMO_ELIGIBLE"
        if 1.5 <= pf <= 2.2:
            return "FOLLOW_ON_OBSERVATION_REQUIRED"
        if pf < 1.0:
            return "SUSPENDED (Negative Expectancy)"
        return "PAPER_ONLY"
        
    if n >= 40:
        if pf > 1.8:
            return "REAL_ELIGIBLE (Subject to Demo Validation)"
        return "FOLLOW_ON_OBSERVATION_REQUIRED"

    return "UNCERTAIN"

def run_tribunal():
    logging.info("=== FORWARD EVIDENCE TRIBUNAL START ===")
    
    if not SCOREBOARD.exists():
        logging.error("Scoreboard not found. Tribunal aborted.")
        return
    
    # 1. Leer Riesgo
    risk_fail = False
    if RISK_STATUS.exists():
        with open(RISK_STATUS, "r") as f:
            risk_data = json.load(f)
            if risk_data.get("results", {}).get("global") == "FAILED":
                risk_fail = True
    
    # 2. Leer Scoreboard
    df = pd.read_csv(SCOREBOARD)
    verdicts = []
    
    for _, row in df.iterrows():
        line = row["Line"]
        verdict = judge_line(row, risk_fail=(risk_fail and line == "SCBI_M5_GLOBAL"))
        verdicts.append({
            "line": line,
            "verdict": verdict,
            "n": int(row["Sample_N"]),
            "pf": float(row["PF_Forward"]),
            "dd": float(row["Max_DD_R"])
        })
        logging.info(f"Verdict for {line}: {verdict}")
        
    # 3. Guardar Resumen
    summary = {
        "timestamp": datetime.now().isoformat(),
        "verdicts": verdicts
    }
    
    TRIBUNAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(TRIBUNAL_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
        
    logging.info("=== FORWARD EVIDENCE TRIBUNAL END ===")
    return summary

if __name__ == "__main__":
    run_tribunal()
