import os
import json
import MetaTrader5 as mt5
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37h_demo_only_terminal_isolation" / "terminal_discovery"

def discover_terminals():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"terminals": []}
    
    # 1. Try currently active terminal via MT5 lib
    if mt5.initialize():
        terminal_info = mt5.terminal_info()
        account_info = mt5.account_info()
        
        info = {
            "source": "MT5_LIB_ACTIVE",
            "path": terminal_info.path if terminal_info else "UNKNOWN",
            "data_path": terminal_info.data_path if terminal_info else "UNKNOWN",
            "company": account_info.company if account_info else "UNKNOWN",
            "server": account_info.server if account_info else "UNKNOWN",
            "login": account_info.login if account_info else 0,
            "trade_mode": account_info.trade_mode if account_info else -1,
            "balance": account_info.balance if account_info else 0,
            "currency": account_info.currency if account_info else "UNKNOWN",
        }
        
        # Classification
        classification = "UNKNOWN_FORBIDDEN"
        if "FTMO" in info["company"] or "FTMO" in info["server"]:
            if info["trade_mode"] in [0, 1, 2]: # DEMO, CONTEST, TRIAL
                classification = "FTMO_DEMO_ALLOWED"
            else:
                classification = "REAL_FORBIDDEN"
        elif "Exness" in info["company"] or "Exness" in info["server"]:
            classification = "EXNESS_FORBIDDEN"
        
        info["classification"] = classification
        report["terminals"].append(info)
        mt5.shutdown()
    else:
        report["terminals"].append({"source": "MT5_LIB_ACTIVE", "error": "Could not initialize"})

    # 2. Check folders in AppData
    terminal_base = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal"
    if terminal_base.exists():
        for d in terminal_base.iterdir():
            if d.is_dir() and len(d.name) == 32: # Typical MT5 hash folder
                origin_file = d / "origin.txt"
                origin = origin_file.read_text().strip() if origin_file.exists() else "UNKNOWN"
                report["terminals"].append({
                    "source": "APPDATA_FOLDER",
                    "hash": d.name,
                    "data_path": str(d),
                    "origin": origin
                })

    with open(OUTPUT_DIR / "phase37h_terminal_discovery.json", "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    discover_terminals()
