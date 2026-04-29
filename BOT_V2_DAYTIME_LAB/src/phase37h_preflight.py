import os
import json
import subprocess
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37h_demo_only_terminal_isolation" / "preflight"

def preflight():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = {
        "cwd": os.getcwd(),
        "exists_manipulante": (ROOT / "MANIPULANTE").exists(),
        "exists_config": (ROOT / "MANIPULANTE" / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "exists_runner": (ROOT / "BOT_V2_DAYTIME_LAB" / "src" / "phase37_ftmo_trial_bot_runner.py").exists(),
        "stop_bot_active": (ROOT / "MANIPULANTE" / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt").exists(),
        "confirmation_file_absent": not (ROOT / "MANIPULANTE" / "13_FTMO_TRIAL_AUTOMATION" / "I_CONFIRM_FTMO_TRIAL_AUTO.txt").exists(),
        "zip_exists": (ROOT / "000_PARA_CHATGPT.zip").exists()
    }
    
    # Git status
    try:
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip()
        checks["git_branch"] = git_branch
    except:
        checks["git_branch"] = "ERROR"

    with open(OUTPUT_DIR / "phase37h_preflight.json", "w") as f:
        json.dump(checks, f, indent=2)
        
    return checks

if __name__ == "__main__":
    preflight()
