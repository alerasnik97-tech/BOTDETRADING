"""
Signal Drift Monitor
Compares forward official evidence against historical baseline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
BASELINE_JSON = ROOT / "results" / "SCBI_SIGNAL_DRIFT_BASELINE.json"
GLOBAL_FWD = ROOT / "results" / "SCBI_FORWARD_LEDGER.csv"
CORE_FWD = ROOT / "results" / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv"

OUTPUT_REPORT = ROOT / "results" / "SCBI_SIGNAL_DRIFT_REPORT.json"

def analyze_drift(df, baseline, line_name):
    if df.empty:
        return {"verdict": "NOT_COMPARABLE_YET", "n": 0}
    
    # Filter for Official
    if line_name == "SCBI_CORE":
        df_off = df[df["event_id"].str.startswith("CORE_")].copy()
    else:
        df_off = df[df["event_type"] == "PAPER_EXIT"].copy()
        
    n = len(df_off)
    if n < 3: # Threshold lower for early stage but still needs data
        return {"verdict": "NOT_COMPARABLE_YET", "n": n}
    
    issues = []
    
    # 1. Composition Check (Levels)
    level_col = "level" if "level" in df_off.columns else "sweep_level"
    fwd_levels = df_off[level_col].value_counts(normalize=True).to_dict()
    hist_levels = baseline["composition"]["levels"]
    
    for lvl, hist_p in hist_levels.items():
        fwd_p = fwd_levels.get(lvl, 0.0)
        if abs(fwd_p - hist_p) > 0.4: # Very loose for small N, tightens as N grows
             issues.append(fwd_p - hist_p)
             
    # 2. Performance Check
    fwd_pnls = df_off["pnl_r"].values
    fwd_exp = float(np.mean(fwd_pnls))
    hist_exp = baseline["performance"]["expectancy"]
    
    verdict = "NO_DRIFT"
    if fwd_exp < (hist_exp - 1.0): # Expectancy drop > 1R
        verdict = "TOLERABLE_VARIATION"
    if fwd_exp < 0 and n > 10:
        verdict = "STRUCTURAL_DRIFT"
        
    return {
        "verdict": verdict,
        "n": n,
        "forward_exp": round(fwd_exp, 4),
        "hist_exp": hist_exp,
        "composition_drift": issues
    }

def main():
    if not BASELINE_JSON.exists():
        print("Baseline not found. Run builder first.")
        return
        
    with open(BASELINE_JSON, "r") as f:
        baselines = json.load(f)
        
    df_global = pd.read_csv(GLOBAL_FWD) if GLOBAL_FWD.exists() else pd.DataFrame()
    df_core = pd.read_csv(CORE_FWD) if CORE_FWD.exists() else pd.DataFrame()
    
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "lines": {}
    }
    
    if "SCBI_M5_GLOBAL" in baselines:
        report["lines"]["SCBI_M5_GLOBAL"] = analyze_drift(df_global, baselines["SCBI_M5_GLOBAL"], "SCBI_M5_GLOBAL")
        
    if "SCBI_CORE" in baselines:
        report["lines"]["SCBI_CORE"] = analyze_drift(df_core, baselines["SCBI_CORE"], "SCBI_CORE")
        
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Drift report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
