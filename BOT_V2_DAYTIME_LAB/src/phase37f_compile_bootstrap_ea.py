import os
import shutil
import subprocess
import json
import time
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MQL5_EA_SOURCE = PROJECT_ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER" / "MANIPULANTE_CalendarBootstrapEA.mq5"
OUTPUT_DIR = PROJECT_ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37f_mql5_calendar_service_exporter" / "compile"

def find_mt5_data_path():
    appdata = os.environ.get("APPDATA")
    if not appdata: return None
    return Path(appdata) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"

def find_metaeditor():
    return Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe")

def compile_ea():
    data_path = find_mt5_data_path()
    metaeditor = find_metaeditor()
    
    mql5_root = data_path / "MQL5"
    target_dir = mql5_root / "Experts" / "MANIPULANTE"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    dest_source = target_dir / MQL5_EA_SOURCE.name
    shutil.copy2(MQL5_EA_SOURCE, dest_source)
    
    log_path = target_dir / "compile.log"
    ps_cmd = f'$argList = "/compile:{dest_source}", "/inc:{mql5_root}", "/log:{log_path}"; Start-Process -FilePath "{metaeditor}" -ArgumentList $argList -Wait'
    
    subprocess.run(["powershell", "-Command", ps_cmd], check=True)
    time.sleep(1)
    
    return dest_source.with_suffix(".ex5").exists()

if __name__ == "__main__":
    if compile_ea():
        print("EA Compiled Successfully")
    else:
        print("EA Compilation Failed")
