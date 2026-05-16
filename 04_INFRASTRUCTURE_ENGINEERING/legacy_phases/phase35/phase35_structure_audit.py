import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'structure_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

findings = []
def add_finding(issue, sev):
    findings.append({'issue': issue, 'severity': sev})

# Check basic
if not (ROOT / 'MANIPULANTE').exists(): add_finding('MANIPULANTE missing', 'BLOCKER')
if not (ROOT / 'ESTRATEGIAS').exists(): add_finding('ESTRATEGIAS missing', 'BLOCKER')
if not (ROOT / 'BOT_V2_DAYTIME_LAB').exists(): add_finding('LAB missing', 'BLOCKER')
if not (ROOT / 'ABRIR_MANIPULANTE_AQUI.txt').exists(): add_finding('ABRIR_MANIPULANTE_AQUI.txt missing', 'WARNING')

# Multiple manipuante folders
count_manip = 0
for p in ROOT.iterdir():
    if p.is_dir() and p.name.lower() == 'manipulante':
        count_manip += 1
if count_manip > 1:
    add_finding('Multiple MANIPULANTE folders', 'BLOCKER')

# Obsolete files in root
obsolete = ['Manipulante', 'BOT DE TRADING', 'legacy']
for p in ROOT.iterdir():
    if p.name in obsolete and p.is_dir() and p.name != 'legacy_archive_2026':
         add_finding(f'Obsolete folder in root: {p.name}', 'WARNING')

if any(f['severity'] == 'BLOCKER' for f in findings):
    verdict = 'BLOCKER'
elif any(f['severity'] == 'WARNING' for f in findings):
    verdict = 'WARNING'
else:
    verdict = 'PASS'

res = {'verdict': verdict, 'findings': findings}

with open(AUDIT_DIR / 'phase35_structure_audit.json', 'w') as f:
    json.dump(res, f, indent=2)

with open(AUDIT_DIR / 'phase35_structure_findings.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['issue', 'severity'])
    writer.writeheader()
    writer.writerows(findings)

md = [f"# STRUCTURE AUDIT\nVerdict: {verdict}\n\n| Issue | Severity |", "|---|---|"]
for f in findings: md.append(f"| {f['issue']} | {f['severity']} |")

with open(AUDIT_DIR / 'phase35_structure_audit.md', 'w') as f:
    f.write('\n'.join(md))

print(json.dumps(res, indent=2))
