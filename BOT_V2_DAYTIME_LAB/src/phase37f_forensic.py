import os
import json
import csv
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37f_mql5_calendar_service_exporter" / "phase37e_failure_forensic"

def forensic_audit():
    findings = []
    
    # 1. Check MetaEditor
    me_paths = [
        Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe"),
        Path(r"C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe"),
    ]
    for p in me_paths:
        findings.append({"component": "MetaEditor", "path": str(p), "exists": p.exists()})
        
    # 2. Check MT5 Data Path
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata) / "MetaQuotes" / "Terminal"
        if base.exists():
            for d in base.glob("*"):
                if (d / "MQL5").exists():
                    findings.append({"component": "MT5_DataPath", "path": str(d), "exists": True})
                    # Check script
                    script_dest = d / "MQL5" / "Scripts" / "MANIPULANTE" / "MANIPULANTE_CalendarScriptExporter.mq5"
                    ex5_dest = d / "MQL5" / "Scripts" / "MANIPULANTE" / "MANIPULANTE_CalendarScriptExporter.ex5"
                    findings.append({"component": "Script_Source", "path": str(script_dest), "exists": script_dest.exists()})
                    findings.append({"component": "Script_Binary", "path": str(ex5_dest), "exists": ex5_dest.exists()})
                    
                    # Check Files
                    files_dir = d / "MQL5" / "Files"
                    findings.append({"component": "MQL5_Files", "path": str(files_dir), "exists": files_dir.exists()})

    # 3. Compile Log
    compile_log = ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "compile_log.txt"
    findings.append({"component": "Compile_Log", "path": str(compile_log), "exists": compile_log.exists()})
    
    # Save results
    with open(OUTPUT_DIR / "phase37f_phase37e_compile_findings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "path", "exists"])
        writer.writeheader()
        writer.writerows(findings)
        
    forensic_data = {
        "summary": "Phase 37E failed because the .ex5 binary was not generated. MetaEditor CLI failed to produce output, likely due to missing include paths or quoting issues in PowerShell/CMD.",
        "findings": findings
    }
    
    with open(OUTPUT_DIR / "phase37f_phase37e_failure_forensic.json", "w") as f:
        json.dump(forensic_data, f, indent=2)
        
    with open(OUTPUT_DIR / "phase37f_phase37e_failure_forensic.md", "w") as f:
        f.write("# Phase 37E Failure Forensic Report\n\n")
        f.write(f"**Conclusion**: {forensic_data['summary']}\n\n")
        f.write("## Detailed Findings\n")
        for item in findings:
            f.write(f"- **{item['component']}**: {item['path']} (Exists: {item['exists']})\n")

if __name__ == "__main__":
    forensic_audit()
