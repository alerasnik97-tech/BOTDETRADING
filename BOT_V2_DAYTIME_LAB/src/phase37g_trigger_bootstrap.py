import os
import subprocess
import time
import MetaTrader5 as mt5
from pathlib import Path

# --- CONFIGURATION ---
TERMINAL = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")
SCRIPT_NAME = "MANIPULANTE_CalendarScriptExporter"

def trigger_script():
    # 1. Ensure MT5 is running
    if not mt5.initialize():
        print("Initializing MT5...")
        subprocess.Popen([str(TERMINAL)])
        time.sleep(15)
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return False

    # 2. Open a chart to ensure there's a target for the script
    chart_id = mt5.chart_open("EURUSD", mt5.TIMEFRAME_H1)
    if chart_id == 0:
        print("Failed to open chart")
        return False
    
    print(f"Opened chart {chart_id}")
    time.sleep(2)
    
    # 3. Trigger script via CLI (which will send it to the running instance)
    cmd = [str(TERMINAL), f"/script:{SCRIPT_NAME}"]
    print(f"Triggering script: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # 4. Wait for cache
    data_path = Path(mt5.terminal_info().data_path)
    cache_file = data_path / "MQL5" / "Files" / "MANIPULANTE" / "ftmo_news_gate_status.json"
    
    start_wait = time.time()
    while time.time() - start_wait < 30:
        if cache_file.exists():
            print("SUCCESS: Cache found!")
            return True
        time.sleep(2)
        
    print("FAILED: Timeout reached")
    return False

if __name__ == "__main__":
    trigger_script()
    mt5.shutdown()
