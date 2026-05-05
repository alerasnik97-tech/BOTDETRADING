import os
import shutil
import subprocess
import time
import glob
import json
from pathlib import Path
from datetime import datetime, timezone

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
TARGET_CACHE_DIR = PROJECT_ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "live_news_cache" / "mql5_script"
MQL5_SOURCE_FILE = PROJECT_ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "MANIPULANTE_CalendarScriptExporter.mq5"

def find_mt5_data_path():
    appdata = os.environ.get("APPDATA")
    if not appdata: return None
    base_path = Path(appdata) / "MetaQuotes" / "Terminal"
    if not base_path.exists(): return None
    terminals = list(base_path.glob("*"))
    if not terminals: return None
    terminals.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    for term in terminals:
        if (term / "MQL5").exists(): return term
    return None

def find_metaeditor():
    # Try common locations
    paths = [
        Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe"),
        Path(r"C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe"),
    ]
    for p in paths:
        if p.exists(): return p
    return None

def find_terminal():
    paths = [
        Path(r"C:\Program Files\MetaTrader 5\terminal64.exe"),
        Path(r"C:\Program Files\FTMO MetaTrader 5\terminal64.exe"),
    ]
    for p in paths:
        if p.exists(): return p
    return None

def compile_script(metaeditor, source_path, data_path):
    print(f"INFO: Compiling {source_path.name}...")
    # Include path is essential
    mql5_root = data_path / "MQL5"
    log_file = PROJECT_ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "compile_log.txt"
    
    # MetaEditor CLI: metaeditor64.exe /compile:source /inc:include_path /log:log_path
    cmd = [
        str(metaeditor),
        f"/compile:{source_path}",
        f"/inc:{mql5_root}",
        f"/log:{log_file}"
    ]
    
    try:
        subprocess.run(cmd, check=False, timeout=30)
        ex5_path = source_path.with_suffix(".ex5")
        if ex5_path.exists():
            print("INFO: Compilation successful.")
            return True
        else:
            print("ERROR: Compilation failed (no .ex5 produced).")
            if log_file.exists():
                try:
                    # Try reading with utf-16
                    content = log_file.read_text(encoding="utf-16")
                    print(f"COMPILER LOG:\n{content}")
                except:
                    pass
            return False
    except Exception as e:
        print(f"ERROR: During compilation subprocess: {e}")
        return False

def execute_script_via_config(terminal, script_name, data_path):
    print(f"INFO: Executing {script_name} via terminal config...")
    config_path = PROJECT_ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "mt5_script_run.ini"
    
    config_content = f"""[Common]
Script={script_name}
"""
    config_path.write_text(config_content)
    
    cmd = [str(terminal), f"/config:{config_path}"]
    try:
        # We start the terminal. If it's already running, it might not pick up the config.
        # So we might need to kill it first if we want 100% automation, 
        # but the user said "no real", so we can be a bit more aggressive.
        # For now, let's just try running it.
        subprocess.Popen(cmd)
        return True
    except Exception as e:
        print(f"ERROR: During execution: {e}")
        return False

def run_automation():
    print("--- PHASE 37E: MQL5 CALENDAR EXPORTER AUTOMATION ---")
    
    data_path = find_mt5_data_path()
    metaeditor = find_metaeditor()
    terminal = find_terminal()
    
    if not data_path or not metaeditor or not terminal:
        print("ERROR: MT5 Environment not detected.")
        return "BLOCKED_MQL5_SCRIPT_AUTORUN_NOT_AVAILABLE"

    scripts_dir = data_path / "MQL5" / "Scripts" / "MANIPULANTE"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    dest_source = scripts_dir / MQL5_SOURCE_FILE.name
    shutil.copy2(MQL5_SOURCE_FILE, dest_source)
    
    if not compile_script(metaeditor, dest_source, data_path):
        return "BLOCKED_MQL5_SCRIPT_AUTORUN_NOT_AVAILABLE"

    script_name = f"MANIPULANTE\\{MQL5_SOURCE_FILE.stem}"
    if not execute_script_via_config(terminal, script_name, data_path):
        return "BLOCKED_MQL5_SCRIPT_AUTORUN_NOT_AVAILABLE"

    print("INFO: Waiting 20 seconds for MT5 to start and run script...")
    time.sleep(20)
    
    mt5_files_dir = data_path / "MQL5" / "Files"
    exported_files = list(mt5_files_dir.glob("MANIPULANTE_news_*.csv"))
    
    if not exported_files:
        print("ERROR: No exported files found in MQL5/Files after execution.")
        return "BLOCKED_MQL5_SCRIPT_AUTORUN_NOT_AVAILABLE"

    TARGET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for f in exported_files:
        print(f"INFO: Copying {f.name} to cache...")
        shutil.copy2(f, TARGET_CACHE_DIR / f.name)
        
    print("SUCCESS: MQL5 Calendar data exported and cached.")
    return "FTMO_TRIAL_AUTO_READY"

if __name__ == "__main__":
    result = run_automation()
    print(f"\nFINAL STATUS: {result}")
    
    status_path = PROJECT_ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "news_automation_status.json"
    with open(status_path, "w") as f:
        json.dump({
            "status": result, 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": "MQL5"
        }, f, indent=2)
