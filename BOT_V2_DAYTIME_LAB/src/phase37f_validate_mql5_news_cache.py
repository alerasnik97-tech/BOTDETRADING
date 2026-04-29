import os
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MT5_DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
MT5_FILES = MT5_DATA_PATH / "MQL5" / "Files" / "MANIPULANTE"
PROJECT_CACHE = PROJECT_ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "live_news_cache" / "mql5_service"
OUTPUT_DIR = PROJECT_ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37f_mql5_calendar_service_exporter" / "cache_validation"

def validate_cache():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
    
    status = {"state": "NO_TRADE_NEWS_CACHE_MISSING", "files": []}
    
    # Try to find files in MT5 first
    targets = ["ftmo_news_today.json", "ftmo_news_week.json", "ftmo_news_today.csv", "ftmo_news_week.csv"]
    
    found_any = False
    for t in targets:
        src = MT5_FILES / t
        if src.exists():
            shutil.copy2(src, PROJECT_CACHE / t)
            status["files"].append(t)
            found_any = True
            
    if not found_any:
        # Check if they already exist in project
        for t in targets:
            if (PROJECT_CACHE / t).exists():
                status["files"].append(t)
                found_any = True

    if found_any:
        # Validate today's JSON
        today_json = PROJECT_CACHE / "ftmo_news_today.json"
        if today_json.exists():
            try:
                with open(today_json, "r") as f:
                    data = json.load(f)
                    
                # Age check
                gen_at_str = data.get("generated_at_utc")
                if gen_at_str:
                    gen_at = datetime.strptime(gen_at_str, "%Y.%m.%d %H:%M:%S")
                    age_minutes = (datetime.utcnow() - gen_at).total_seconds() / 60
                    status["cache_age_minutes"] = age_minutes
                    
                    if age_minutes <= 60:
                        status["state"] = "LIVE_NEWS_CACHE_VALID"
                    else:
                        status["state"] = "NO_TRADE_NEWS_CACHE_STALE"
            except Exception as e:
                status["state"] = "ERROR_PARSING"
                status["details"] = str(e)

    with open(OUTPUT_DIR / "phase37f_cache_validation.json", "w") as f:
        json.dump(status, f, indent=2)
        
    return status

if __name__ == "__main__":
    validate_cache()
