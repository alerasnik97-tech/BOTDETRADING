import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'safety_gates_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

findings = []

config_path = ROOT / 'MANIPULANTE' / '01_ESTRATEGIA_AUTORIDAD' / 'manipulante_config.json'
with open(config_path, 'r') as f:
    cfg = json.load(f)

if cfg.get("news_fortress") == "FAIL_CLOSED_REQUIRED":
    findings.append({"gate": "News Fortress", "status": "FAIL_CLOSED", "severity": "PASS"})
else:
    findings.append({"gate": "News Fortress", "status": "UNCERTAIN", "severity": "BLOCKER"})

if cfg.get("data_quality_mask") == "FAIL_CLOSED_REQUIRED":
    findings.append({"gate": "Data Quality Mask", "status": "FAIL_CLOSED", "severity": "PASS"})
else:
    findings.append({"gate": "Data Quality Mask", "status": "UNCERTAIN", "severity": "BLOCKER"})

res = {"verdict": "PASS" if not any(f['severity'] == "BLOCKER" for f in findings) else "BLOCKER", "findings": findings}

with open(AUDIT_DIR / 'phase35_safety_gates_audit.json', 'w') as f:
    json.dump(res, f, indent=2)

md = ["# SAFETY GATES AUDIT\n\n| Gate | Status | Severity |", "|---|---|---|"]
for f in findings:
    md.append(f"| {f['gate']} | {f['status']} | {f['severity']} |")

with open(AUDIT_DIR / 'phase35_safety_gates_audit.md', 'w') as f:
    f.write('\n'.join(md))

print("Safety gates audit completed.")
