import os
import subprocess
import time
import json
from pathlib import Path

# --- CONFIGURATION ---
TERMINAL = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")
DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
GATE_STATUS = DATA_PATH / "MQL5" / "Files" / "MANIPULANTE" / "ftmo_news_gate_status.json"
OUTPUT_DIR = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase37h_demo_only_terminal_isolation\ftmo_mql5_bootstrap_resume")

def resume_bootstrap():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {"state": "RESUME_STARTED"}
    
    # 1. Start MT5
    subprocess.Popen([str(TERMINAL)])
    print("MT5 started. Waiting for bootstrap...")
    
    # 2. Wait for gate status
    found = False
    start_wait = time.time()
    timeout = 120 # Give it 2 minutes
    
    while time.time() - start_wait < timeout:
        if GATE_STATUS.exists():
            try:
                with open(GATE_STATUS, "r") as f:
                    data = json.load(f)
                    if data.get("status") == "BOOTSTRAP_DONE":
                        found = True
                        status["state"] = "FTMO_MQL5_CACHE_CREATED"
                        status["today_count"] = data.get("today_count")
                        break
            except: pass
        time.sleep(5)
        
    if not found:
        status["state"] = "BLOCKED_MQL5_AUTOSTART"
        
    with open(OUTPUT_DIR / "phase37h_ftmo_mql5_bootstrap_resume.json", "w") as f:
        json.dump(status, f, indent=2)
        
    return status

if __name__ == "__main__":
    resume_bootstrap()
