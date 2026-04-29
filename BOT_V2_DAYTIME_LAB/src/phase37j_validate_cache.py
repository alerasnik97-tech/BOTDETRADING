import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

# --- CONFIGURATION ---
DATA_PATH = Path(os.environ.get("APPDATA")) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075"
MT5_FILES = DATA_PATH / "MQL5" / "Files" / "MANIPULANTE"
PROJECT_CACHE = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\09_COMPLIANCE\live_news_cache\mql5_bootstrap")
OUTPUT_DIR = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase37j_post_bootstrap_validation\news_cache_validation")

def validate_cache():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
    
    report = {"state": "NO_TRADE_NEWS_CACHE_MISSING", "files": []}
    
    targets = ["ftmo_news_today.json", "ftmo_news_week.json", "ftmo_news_gate_status.json"]
    
    found_any = False
    for t in targets:
        src = MT5_FILES / t
        if src.exists():
            # Copy to project
            shutil.copy2(src, PROJECT_CACHE / t)
            report["files"].append(t)
            found_any = True

    if found_any:
        # Validate today's JSON
        today_json = PROJECT_CACHE / "ftmo_news_today.json"
        if today_json.exists():
            try:
                with open(today_json, "r") as f:
                    data = json.load(f)
                
                report["source"] = data.get("source")
                gen_at_str = data.get("generated_at_utc")
                report["generated_at"] = gen_at_str
                
                if gen_at_str:
                    # Format: 2026.04.29 11:21:00
                    gen_at = datetime.strptime(gen_at_str, "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    age_seconds = (datetime.now(timezone.utc) - gen_at).total_seconds()
                    report["age_minutes"] = age_seconds / 60
                    
                    if age_seconds <= 3600: # 1 hour
                        report["state"] = "LIVE_NEWS_CACHE_VALID"
                    else:
                        report["state"] = "NO_TRADE_NEWS_CACHE_STALE"
                
                report["events_count"] = len(data.get("events", []))
            except Exception as e:
                report["state"] = "ERROR_PARSING"
                report["error"] = str(e)

    with open(OUTPUT_DIR / "phase37j_news_cache_validation.json", "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    validate_cache()
