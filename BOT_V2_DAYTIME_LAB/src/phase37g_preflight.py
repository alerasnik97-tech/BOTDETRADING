import os
import json
import MetaTrader5 as mt5
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37g_mql5_news_auto_bootstrap" / "preflight"

def preflight():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = {}
    
    # Check MT5 connection
    if mt5.initialize():
        checks["mt5_initialized"] = True
        acc = mt5.account_info()
        if acc:
            checks["account"] = acc.login
            checks["company"] = acc.company
            checks["mode"] = acc.trade_mode
        mt5.shutdown()
    else:
        checks["mt5_initialized"] = False
        checks["mt5_error"] = mt5.last_error()

    # Check existence of EA
    data_path = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
    ea_path = data_path / "MQL5" / "Experts" / "MANIPULANTE" / "MANIPULANTE_CalendarBootstrapEA.ex5"
    checks["ea_exists"] = ea_path.exists()
    
    # Check current cache
    cache_file = data_path / "MQL5" / "Files" / "MANIPULANTE" / "ftmo_news_gate_status.json"
    checks["cache_exists"] = cache_file.exists()

    with open(OUTPUT_DIR / "phase37g_preflight.json", "w") as f:
        json.dump(checks, f, indent=2)
        
    return checks

if __name__ == "__main__":
    preflight()
