import os
import json
import re
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MQL5_SOURCE = PROJECT_ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER" / "MANIPULANTE_CalendarServiceExporter.mq5"
OUTPUT_DIR = PROJECT_ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37f_mql5_calendar_service_exporter" / "service_static_audit"

def static_audit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not MQL5_SOURCE.exists():
        return {"error": "Source not found"}
        
    code = MQL5_SOURCE.read_text(encoding="utf-8")
    
    forbidden = [
        r"OrderSend", r"CTrade", r"trade\.", r"Buy\(", r"Sell\(", 
        r"PositionOpen", r"PositionClose", r"OrderSendAsync",
        r"HistoryOrder", r"ImportDll", r"WebRequest"
    ]
    
    findings = []
    for pattern in forbidden:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            findings.append({"pattern": pattern, "count": len(matches)})
            
    # Key functions that SHOULD be there
    required = ["CalendarValueHistory", "CalendarEventById", "CalendarCountryById", "FileWrite", "OnStart"]
    missing = [f for f in required if f not in code]
    
    status = {
        "state": "PASS" if not findings else "BLOCKER",
        "findings": findings,
        "missing_required": missing
    }
    
    with open(OUTPUT_DIR / "phase37f_service_static_audit.json", "w") as f:
        json.dump(status, f, indent=2)
        
    with open(OUTPUT_DIR / "phase37f_service_static_audit.md", "w") as f:
        f.write("# Phase 37F Service Static Safety Audit\n\n")
        f.write(f"**Veredict**: {status['state']}\n\n")
        f.write("## Findings\n")
        if findings:
            for find in findings:
                f.write(f"- ❌ Found forbidden pattern `{find['pattern']}`: {find['count']} times\n")
        else:
            f.write("- ✅ No forbidden trading functions found.\n")
            
    return status

if __name__ == "__main__":
    static_audit()
