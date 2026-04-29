import os
import shutil
import json
import subprocess
import time
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
COMMON_INI = DATA_PATH / "config" / "common.ini"
CHART_FILE = DATA_PATH / "MQL5" / "Profiles" / "Charts" / "default" / "chart01.chr"
OUTPUT_DIR = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase37h_demo_only_terminal_isolation\ftmo_recovery")

def restore_ftmo_env():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {"state": "INIT"}
    
    # 1. Kill MT5 first
    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], check=False)
    time.sleep(2)

    # 2. Restore common.ini (Enable Algo Trading)
    if COMMON_INI.exists():
        shutil.copy2(COMMON_INI, OUTPUT_DIR / "common.ini.bak")
        content = COMMON_INI.read_bytes().decode("utf-16", errors="ignore")
        if "Enabled=0" in content:
            new_content = content.replace("Enabled=0", "Enabled=1")
            COMMON_INI.write_bytes(new_content.encode("utf-16"))
            status["common_ini"] = "RESTORED_ENABLED_1"
        else:
            status["common_ini"] = "ALREADY_ENABLED"
            
    # 3. Restore chart01.chr (Inject EA)
    if CHART_FILE.exists():
        shutil.copy2(CHART_FILE, OUTPUT_DIR / "chart01.chr.bak")
        content = CHART_FILE.read_bytes().decode("utf-16", errors="ignore")
        if "MANIPULANTE_CalendarBootstrapEA" not in content:
            expert_section = """
<expert>
name=MANIPULANTE\\\\MANIPULANTE_CalendarBootstrapEA
flags=339
window=0
<inputs>
</inputs>
</expert>
"""
            new_content = content.replace("</chart>", expert_section + "</chart>")
            CHART_FILE.write_bytes(new_content.encode("utf-16"))
            status["chart_file"] = "INJECTED_EA"
        else:
            status["chart_file"] = "ALREADY_INJECTED"

    status["state"] = "FTMO_ENV_RECOVERED"
    
    with open(OUTPUT_DIR / "phase37h_ftmo_recovery.json", "w") as f:
        json.dump(status, f, indent=2)
        
    return status

if __name__ == "__main__":
    restore_ftmo_env()
