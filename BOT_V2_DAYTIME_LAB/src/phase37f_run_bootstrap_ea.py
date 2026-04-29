import os
import subprocess
import time
import json
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OUTPUT_DIR = PROJECT_ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37f_mql5_calendar_service_exporter" / "service_autostart"

def find_mt5_data_path():
    appdata = os.environ.get("APPDATA")
    if not appdata: return None
    return Path(appdata) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"

def find_terminal():
    return Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")

def run_bootstrap_ea_v2():
    terminal = find_terminal()
    data_path = find_mt5_data_path()
    
    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], check=False)
    time.sleep(2)
    
    # Try different section name [Experts] and also ensure it's on a chart
    config_path = PROJECT_ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "mt5_bootstrap_ea.ini"
    config_content = f"""[Common]
Login=1513250440
[Charts]
Symbol=EURUSD
Period=H1
[Experts]
AllowLiveTrading=0
AllowDllImport=1
Enabled=1
Expert=MANIPULANTE\\MANIPULANTE_CalendarBootstrapEA
"""
    config_path.write_text(config_content)
    
    cmd = [str(terminal), f"/config:{config_path}"]
    print(f"INFO: Starting MT5 with Bootstrap EA V2: {' '.join(cmd)}")
    
    subprocess.Popen(cmd)
    
    gate_status_file = data_path / "MQL5" / "Files" / "MANIPULANTE" / "ftmo_news_gate_status.json"
    start_wait = time.time()
    timeout = 60
    found = False
    
    while time.time() - start_wait < timeout:
        if gate_status_file.exists():
            try:
                with open(gate_status_file, "r") as f:
                    data = json.load(f)
                    if data.get("status") in ["BOOTSTRAP_DONE", "RUNNING"]:
                        found = True
                        break
            except: pass
        time.sleep(2)
        
    return found

if __name__ == "__main__":
    if run_bootstrap_ea_v2():
        print("SUCCESS: Cache Created")
    else:
        print("FAILED: No Cache Created")
