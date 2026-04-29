import os, json, csv, subprocess, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'repo_zip_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

findings = []

# Zip scan
zip_path = ROOT / '000_PARA_CHATGPT.zip'
if not zip_path.exists():
    findings.append({"item": "ZIP", "status": "NOT_FOUND", "severity": "BLOCKER"})
else:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            with open(AUDIT_DIR / 'phase35_zip_entries.txt', 'w') as f:
                f.write('\n'.join(names))
            
            banned = ['.env', 'secret', 'credentials', 'password']
            for name in names:
                if any(b in name.lower() for b in banned):
                    findings.append({"item": f"ZIP_ENTRY:{name}", "status": "BANNED_FILE_DETECTED", "severity": "BLOCKER"})
    except:
        findings.append({"item": "ZIP", "status": "CORRUPT", "severity": "BLOCKER"})

# Git status
try:
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    if ".env" in status or "secret" in status.lower():
        findings.append({"item": "GIT_STATUS", "status": "SECRET_DETECTED", "severity": "BLOCKER"})
except: pass

res = {"verdict": "PASS" if not any(f['severity'] == "BLOCKER" for f in findings) else "BLOCKER", "findings": findings}

with open(AUDIT_DIR / 'phase35_repo_zip_audit.json', 'w') as f:
    json.dump(res, f, indent=2)

md = ["# REPO AND ZIP AUDIT\n\n| Item | Status | Severity |", "|---|---|---|"]
for f in findings:
    md.append(f"| {f['item']} | {f['status']} | {f['severity']} |")

with open(AUDIT_DIR / 'phase35_repo_zip_audit.md', 'w') as f:
    f.write('\n'.join(md))

print("Repo and ZIP audit completed.")
