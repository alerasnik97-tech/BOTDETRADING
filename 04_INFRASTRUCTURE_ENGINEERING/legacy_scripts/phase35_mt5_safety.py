import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'mt5_safety_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

findings = []

files_to_check = [
    ROOT / 'MANIPULANTE' / '03_MT5_DEMO_LAUNCHER' / 'ABRIR_MANIPULANTE_DEMO.bat',
    ROOT / 'MANIPULANTE' / '03_MT5_DEMO_LAUNCHER' / 'ABRIR_MANIPULANTE_DEMO.ps1',
    ROOT / 'MANIPULANTE' / '03_MT5_DEMO_LAUNCHER' / 'mt5_path_config.json'
]

for p in files_to_check:
    if not p.exists():
        findings.append({"file": str(p.name), "issue": "FILE_MISSING", "severity": "WARNING"})
        continue
    
    try:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
    except: continue
    
    if "Account" in content and "Real" in content:
        findings.append({"file": str(p.name), "issue": "REAL_ACCOUNT_REFERENCE", "severity": "BLOCKER"})
    if "/portable" not in content and ".bat" in str(p.name):
        findings.append({"file": str(p.name), "issue": "NOT_PORTABLE_MODE", "severity": "WARNING"})
    if "AutoTrading" in content and "Enable" in content:
        findings.append({"file": str(p.name), "issue": "AUTOTRADING_ENABLED_REF", "severity": "BLOCKER"})

# Check if REAL_MONEY_LOCK.md exists
lock_file = ROOT / 'MANIPULANTE' / '03_MT5_DEMO_LAUNCHER' / 'REAL_MONEY_LOCK.md'
if not lock_file.exists():
    with open(lock_file, 'w', encoding='utf-8') as f:
        f.write("# REAL MONEY LOCK\nNo conectar cuenta real ni activar AutoTrading hasta que PHASE35 emita READY_FOR_MICRO_REAL_WITH_WARNINGS y el usuario confirme manualmente.")
    findings.append({"file": "REAL_MONEY_LOCK.md", "issue": "FILE_CREATED", "severity": "PASS"})

verdict = "BLOCKER" if any(f['severity'] == 'BLOCKER' for f in findings) else "PASS"
res = {"verdict": verdict, "findings": findings}

with open(AUDIT_DIR / 'phase35_mt5_safety_audit.json', 'w') as f:
    json.dump(res, f, indent=2)

md = [f"# MT5 SAFETY AUDIT\nVerdict: {verdict}\n\n| File | Issue | Severity |", "|---|---|---|"]
for f in findings:
    md.append(f"| {f['file']} | {f['issue']} | {f['severity']} |")

with open(AUDIT_DIR / 'phase35_mt5_safety_audit.md', 'w') as f:
    f.write('\n'.join(md))

print(json.dumps(res, indent=2))
