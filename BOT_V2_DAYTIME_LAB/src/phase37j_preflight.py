import os
import json
import subprocess
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37j_post_bootstrap_validation" / "preflight"

def preflight():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = {
        "cwd": os.getcwd(),
        "exists_manipulante": (ROOT / "MANIPULANTE").exists(),
        "exists_config": (ROOT / "MANIPULANTE" / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "exists_runner": (ROOT / "BOT_V2_DAYTIME_LAB" / "src" / "phase37_ftmo_trial_bot_runner.py").exists(),
        "stop_bot_active": (ROOT / "MANIPULANTE" / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt").exists(),
        "zip_exists": (ROOT / "000_PARA_CHATGPT.zip").exists()
    }
    
    # Check for Bootstrap EA binary
    data_path = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
    checks["bootstrap_ea_exists"] = (data_path / "MQL5" / "Experts" / "MANIPULANTE" / "MANIPULANTE_CalendarBootstrapEA.ex5").exists()

    with open(OUTPUT_DIR / "phase37j_preflight.json", "w") as f:
        json.dump(checks, f, indent=2)
        
    return checks

if __name__ == "__main__":
    preflight()
