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

def run_bootstrap_with_kill():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {"state": "INIT"}
    
    terminal = find_terminal()
    data_path = find_mt5_data_path()
    
    # 1. Kill MT5
    try:
        subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], check=False, capture_output=True)
        time.sleep(2)
    except:
        pass
        
    # 2. Create config file
    config_path = PROJECT_ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "mt5_bootstrap_config.ini"
    config_content = f"""[Common]
Script=MANIPULANTE\\MANIPULANTE_CalendarScriptExporter
"""
    config_path.write_text(config_content)
    
    # 3. Start MT5
    cmd = [str(terminal), f"/config:{config_path}"]
    print(f"INFO: Starting MT5 with Config: {' '.join(cmd)}")
    
    try:
        subprocess.Popen(cmd)
        
        # 4. Wait for cache
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
                except:
                    pass
            time.sleep(2)
            
        if found:
            status["state"] = "SERVICE_RUNNING_CACHE_CREATED"
        else:
            status["state"] = "BLOCKED_NO_CACHE_CREATED"
            # Check for errors in logs
            log_dir = data_path / "logs"
            today_log = log_dir / (time.strftime("%Y%m%d") + ".log")
            if today_log.exists():
                status["details"] = "Check logs: " + str(today_log)
                
    except Exception as e:
        status["state"] = "ERROR_SUBPROCESS"
        status["details"] = str(e)

    with open(OUTPUT_DIR / "phase37f_service_autostart.json", "w") as f:
        json.dump(status, f, indent=2)
            
    return status

if __name__ == "__main__":
    run_bootstrap_with_kill()
