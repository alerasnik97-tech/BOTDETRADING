import os
import json
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
OUTPUT_DIR = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase37h_demo_only_terminal_isolation\safety_recovery_audit")

def safety_audit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    findings = {}
    
    # Check common.ini for Enabled=0 (Algo Trading)
    common_ini = DATA_PATH / "config" / "common.ini"
    if common_ini.exists():
        content = common_ini.read_bytes().decode("utf-16", errors="ignore")
        findings["algo_trading_enabled"] = "Enabled=1" in content
        findings["algo_trading_disabled"] = "Enabled=0" in content
    else:
        findings["common_ini"] = "MISSING"

    # Check chart01.chr for EA injection
    chart_file = DATA_PATH / "MQL5" / "Profiles" / "Charts" / "default" / "chart01.chr"
    if chart_file.exists():
        content = chart_file.read_bytes().decode("utf-16", errors="ignore")
        findings["ea_in_chart"] = "MANIPULANTE_CalendarBootstrapEA" in content
    else:
        findings["chart_file"] = "MISSING"

    # Check for compiled binaries
    mql5_root = DATA_PATH / "MQL5"
    findings["service_ex5_exists"] = (mql5_root / "Services" / "MANIPULANTE" / "MANIPULANTE_CalendarServiceExporter.ex5").exists()
    findings["bootstrap_ex5_exists"] = (mql5_root / "Experts" / "MANIPULANTE" / "MANIPULANTE_CalendarBootstrapEA.ex5").exists()

    with open(OUTPUT_DIR / "phase37h_safety_recovery_audit.json", "w") as f:
        json.dump(findings, f, indent=2)
        
    return findings

if __name__ == "__main__":
    safety_audit()
