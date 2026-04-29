import os
import shutil
import subprocess
import json
import time
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MQL5_SERVICE_SOURCE = PROJECT_ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER" / "MANIPULANTE_CalendarServiceExporter.mq5"
MQL5_SCRIPT_SOURCE = PROJECT_ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "MANIPULANTE_CalendarScriptExporter.mq5"
OUTPUT_DIR = PROJECT_ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37f_mql5_calendar_service_exporter" / "compile"

def find_mt5_data_path():
    appdata = os.environ.get("APPDATA")
    if not appdata: return None
    base = Path(appdata) / "MetaQuotes" / "Terminal"
    if not base.exists(): return None
    d = base / "D0E8209F77C8CF37AD8BF550E51FF075"
    if d.exists(): return d
    return None

def find_metaeditor():
    return Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe")

def compile_mql5(metaeditor, source_path, data_path, target_type):
    mql5_root = data_path / "MQL5"
    target_dir = mql5_root / target_type / "MANIPULANTE"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    dest_source = target_dir / source_path.name
    shutil.copy2(source_path, dest_source)
    
    log_path = target_dir / "compile.log"
    if log_path.exists(): log_path.unlink()
    
    ps_cmd = f'$argList = "/compile:{dest_source}", "/inc:{mql5_root}", "/log:{log_path}"; Start-Process -FilePath "{metaeditor}" -ArgumentList $argList -Wait'
    
    try:
        subprocess.run(["powershell", "-Command", ps_cmd], check=True, timeout=90)
        time.sleep(1)
        dest_ex5 = dest_source.with_suffix(".ex5")
        if dest_ex5.exists():
            return {"state": "OK", "ex5": str(dest_ex5)}
        else:
            details = "No log"
            if log_path.exists():
                try:
                    details = log_path.read_bytes().decode("utf-16")
                except: pass
            return {"state": "FAILED", "details": details}
    except Exception as e:
        return {"state": "ERROR", "details": str(e)}

def run_compilations():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = find_mt5_data_path()
    metaeditor = find_metaeditor()
    
    results = {}
    if data_path and metaeditor:
        results["service"] = compile_mql5(metaeditor, MQL5_SERVICE_SOURCE, data_path, "Services")
        results["script"] = compile_mql5(metaeditor, MQL5_SCRIPT_SOURCE, data_path, "Scripts")
    else:
        results["error"] = "MT5 or MetaEditor not found"

    with open(OUTPUT_DIR / "phase37f_compile_service.json", "w") as f:
        json.dump(results, f, indent=2)
            
    return results

if __name__ == "__main__":
    run_compilations()
