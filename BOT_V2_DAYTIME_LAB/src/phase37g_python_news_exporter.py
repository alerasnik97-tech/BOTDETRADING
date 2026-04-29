import requests
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
CACHE_DIR = PROJECT_ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "live_news_cache" / "mql5_service"

def export_news_ff():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}"}
            
        data = response.json()
        events = []
        for item in data:
            if item["country"] not in ["EUR", "USD"]: continue
            if item["impact"] not in ["High", "Medium"]: continue
            
            # Date: 2026-04-26T16:00:00-04:00
            try:
                dt_obj = datetime.fromisoformat(item["date"])
                t_utc = dt_obj.astimezone(timezone.utc)
                t_str = t_utc.strftime("%Y.%m.%d %H:%M")
            except:
                continue
            
            imp = "HIGH" if item["impact"] == "High" else "MODERATE"
            
            events.append({
                "name": item["title"],
                "currency": item["country"],
                "country": item["country"],
                "importance": imp,
                "time_utc": t_str
            })
            
        now_str = datetime.now(timezone.utc).strftime("%Y.%m.%d %H:%M:%S")
        payload = {
            "source": "FOREX_FACTORY_HYBRID_BRIDGE",
            "generated_at_utc": now_str,
            "events": events
        }
        
        with open(CACHE_DIR / "ftmo_news_today.json", "w") as f:
            json.dump(payload, f, indent=2)
        with open(CACHE_DIR / "ftmo_news_week.json", "w") as f:
            json.dump(payload, f, indent=2)
            
        with open(CACHE_DIR / "ftmo_news_gate_status.json", "w") as f:
            json.dump({
                "status": "RUNNING",
                "last_update_utc": now_str,
                "today_count": len(events)
            }, f, indent=2)
            
        return {"status": "SUCCESS", "count": len(events)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    res = export_news_ff()
    print(json.dumps(res, indent=2))
