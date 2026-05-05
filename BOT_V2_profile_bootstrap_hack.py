import os
import subprocess
import time
import json
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
CHART_FILE = DATA_PATH / "MQL5" / "Profiles" / "Charts" / "default" / "chart01.chr"
TERMINAL = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")

def inject_ea_to_chart():
    if not CHART_FILE.exists():
        print(f"Chart file not found: {CHART_FILE}")
        return False
        
    # Read UTF-16
    content = CHART_FILE.read_bytes().decode("utf-16")
    
    # Check if EA already there
    if "MANIPULANTE_CalendarBootstrapEA" in content:
        print("EA already in chart configuration.")
    else:
        # Inject <expert> section before the last </chart>
        expert_section = """
<expert>
name=MANIPULANTE\\MANIPULANTE_CalendarBootstrapEA
flags=339
window=0
<inputs>
</inputs>
</expert>
"""
        new_content = content.replace("</chart>", expert_section + "</chart>")
        
        # Backup and Write
        shutil_copy = Path(str(CHART_FILE) + ".bak")
        CHART_FILE.rename(shutil_copy)
        CHART_FILE.write_bytes(new_content.encode("utf-16"))
        print("EA injected into chart01.chr")

    # Restart MT5
    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], check=False)
    time.sleep(2)
    subprocess.Popen([str(TERMINAL)])
    print("MT5 restarted. EA should load automatically.")
    
    # Wait for cache
    gate_status = DATA_PATH / "MQL5" / "Files" / "MANIPULANTE" / "ftmo_news_gate_status.json"
    start_wait = time.time()
    while time.time() - start_wait < 60:
        if gate_status.exists():
            try:
                with open(gate_status, "r") as f:
                    data = json.load(f)
                    if data.get("status") == "BOOTSTRAP_DONE":
                        print("SUCCESS: Bootstrap hack worked!")
                        return True
            except: pass
        time.sleep(2)
        
    return False

if __name__ == "__main__":
    inject_ea_to_chart()
