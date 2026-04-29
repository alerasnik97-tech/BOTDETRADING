import os
import json
import hashlib
import zipfile
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MANIPULANTE = ROOT / "MANIPULANTE"
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUTPUT_DIR = LAB / "outputs" / "phase37f_mql5_calendar_service_exporter" / "preflight"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"

def get_sha256(file_path):
    if not file_path.exists(): return None
    sha256_hash = hashlib.sha256()
    with open(file_path,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def preflight_check():
    checks = {}
    
    # Paths
    checks["cwd"] = str(os.getcwd())
    checks["root_exists"] = ROOT.exists()
    checks["manipulante_exists"] = MANIPULANTE.exists()
    checks["lab_exists"] = LAB.exists()
    
    # Config
    config_path = MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json"
    checks["manipulante_config_exists"] = config_path.exists()
    
    # Phase37E Status
    status_37e = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "PHASE37E_STATUS.md"
    checks["phase37e_status_exists"] = status_37e.exists()
    
    # Runner
    runner_path = LAB / "src" / "phase37_ftmo_trial_bot_runner.py"
    checks["bot_runner_exists"] = runner_path.exists()
    
    # ZIP
    checks["zip_exists"] = ZIP_PATH.exists()
    if checks["zip_exists"]:
        checks["zip_sha256"] = get_sha256(ZIP_PATH)
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
                checks["zip_test"] = zf.testzip() is None
                checks["zip_entries"] = len(zf.namelist())
        except:
            checks["zip_test"] = False
            
    # Account / MT5 (Briefly check module)
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            acc = mt5.account_info()
            checks["mt5_connected"] = acc is not None
            if acc:
                checks["account_mode"] = acc.trade_mode
                checks["account_company"] = acc.company
                checks["account_balance"] = acc.balance
            mt5.shutdown()
        else:
            checks["mt5_connected"] = False
    except:
        checks["mt5_connected"] = False

    # Safety
    stop_bot = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "STOP_BOT.txt"
    checks["stop_bot_active"] = stop_bot.exists()
    
    conf_file = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "I_CONFIRM_FTMO_TRIAL_AUTO.txt"
    checks["confirmation_file_exists"] = conf_file.exists()

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "phase37f_preflight.json", "w") as f:
        json.dump(checks, f, indent=2)
        
    with open(OUTPUT_DIR / "phase37f_preflight.md", "w") as f:
        f.write("# Phase 37F Preflight Check\n\n")
        for k, v in checks.items():
            f.write(f"- **{k}**: {v}\n")
            
    return checks

if __name__ == "__main__":
    preflight_check()
