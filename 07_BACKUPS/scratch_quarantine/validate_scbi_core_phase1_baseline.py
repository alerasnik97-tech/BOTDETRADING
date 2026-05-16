"""
SCBI_CORE Phase 1 Baseline Freezer

Takes the Full Campaign results and freezes them as the official Phase 1 baseline.
Ensures drift monitoring has a solid, namespaced starting point.
"""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
INPUT_METRICS = ROOT / "results" / "SCBI_CORE_FULL_CAMPAIGN" / "core_full_campaign_metrics.json"
OUTPUT_FILE = ROOT / "results" / "SCBI_CORE_PHASE1" / "core_phase1_frozen_baseline.json"

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(INPUT_METRICS, "r") as f:
        data = json.load(f)
    
    # Extract Global Baseline metrics (Institutional 0.4 pips)
    baseline = data["baseline_04"]["global"]
    
    frozen = {
        "strategy": "SCBI_CORE",
        "freeze_date": "2026-04-22",
        "historical_n": baseline["N"],
        "historical_pf": baseline["pf"],
        "historical_expectancy": baseline["expectancy"],
        "historical_max_dd": baseline["max_dd"],
        "segments": {
            "dev_pf": data["baseline_04"]["blocks"]["Development"]["pf"],
            "val_pf": data["baseline_04"]["blocks"]["Validation"]["pf"],
            "holdout_pf": data["baseline_04"]["blocks"]["Holdout"]["pf"]
        },
        "target_drift_limit": 0.5 # 50% max degradation
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(frozen, f, indent=2)
    
    print(f"Phase 1 Baseline Frozen for SCBI_CORE: PF {baseline['pf']}")

if __name__ == "__main__":
    main()
