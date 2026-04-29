import os
import json
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
OUTPUT_DIR = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase37j_post_bootstrap_validation\mql5_component_detection")

def detect_components():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"components": []}
    
    # 1. Search in Chart Profiles (to see if EAs are attached)
    profile_dir = DATA_PATH / "MQL5" / "Profiles" / "Charts" / "default"
    if profile_dir.exists():
        for f in profile_dir.glob("*.chr"):
            content = f.read_bytes().decode("utf-16", errors="ignore")
            if "MANIPULANTE_CalendarBootstrapEA" in content:
                report["components"].append({"name": "MANIPULANTE_CalendarBootstrapEA", "state": "ATTACHED_TO_CHART", "file": f.name})
            if "MANIPULANTE_CalendarBridgeEA" in content:
                report["components"].append({"name": "MANIPULANTE_CalendarBridgeEA", "state": "ATTACHED_TO_CHART_WARNING", "file": f.name})

    # 2. Search in Experts Log
    log_dir = DATA_PATH / "MQL5" / "Logs"
    today_log = log_dir / (datetime.utcnow().strftime("%Y%m%d") + ".log")
    # We'll check the most recent log file
    logs = sorted(log_dir.glob("*.log"))
    if logs:
        latest_log = logs[-1]
        try:
            log_content = latest_log.read_bytes().decode("utf-16", errors="ignore")
            if "MANIPULANTE_CalendarBootstrapEA" in log_content:
                report["components"].append({"name": "MANIPULANTE_CalendarBootstrapEA", "state": "DETECTED_IN_LOGS", "log": latest_log.name})
            if "MANIPULANTE_CalendarBridgeEA" in log_content:
                report["components"].append({"name": "MANIPULANTE_CalendarBridgeEA", "state": "DETECTED_IN_LOGS", "log": latest_log.name})
        except: pass

    with open(OUTPUT_DIR / "phase37j_mql5_component_detection.json", "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    from datetime import datetime
    detect_components()
