import os
import subprocess
import time
import json
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
TERMINAL = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")
DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"

def run_bootstrap_v3():
    # Kill MT5
    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], check=False)
    time.sleep(2)
    
    # Create config in a simple path
    config_path = PROJECT_ROOT / "bootstrap.ini"
    config_content = f"""[Common]
Login=1513250440
ProxyEnable=0
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
    
    # Start MT5 with explicit login and config
    cmd = [str(TERMINAL), "/login:1513250440", "/server:FTMO-Demo", f"/config:{config_path}"]
    print(f"INFO: Executing: {' '.join(cmd)}")
    subprocess.Popen(cmd)
    
    # Wait for heartbeat
    heartbeat = DATA_PATH / "MQL5" / "Files" / "MANIPULANTE" / "ea_init.tmp"
    gate_status = DATA_PATH / "MQL5" / "Files" / "MANIPULANTE" / "ftmo_news_gate_status.json"
    
    start_wait = time.time()
    while time.time() - start_wait < 60:
        if heartbeat.exists():
            print("INFO: EA Heartbeat detected!")
        if gate_status.exists():
            try:
                with open(gate_status, "r") as f:
                    data = json.load(f)
                    if data.get("status") == "BOOTSTRAP_DONE":
                        print("SUCCESS: Bootstrap Completed!")
                        return True
            except: pass
        time.sleep(2)
        
    print("FAILED: Timeout")
    return False

if __name__ == "__main__":
    run_bootstrap_v3()
