import os
import json
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MQL5_DIR = ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER"
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37h_demo_only_terminal_isolation" / "mql5_safety_revalidation"

FORBIDDEN_PATTERNS = [
    "OrderSend", "CTrade", "PositionOpen", "Buy(", "Sell(", "OrderCheck", "trade.",
    "DllImport", "WebRequest", "password", "token"
]

def audit_mql5():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"files": []}
    
    for f in MQL5_DIR.glob("*.mq5"):
        content = f.read_text(errors="ignore")
        findings = []
        for p in FORBIDDEN_PATTERNS:
            if p in content:
                findings.append(p)
        
        report["files"].append({
            "name": f.name,
            "findings": findings,
            "pass": len(findings) == 0
        })

    with open(OUTPUT_DIR / "phase37h_mql5_safety_revalidation.json", "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    audit_mql5()
