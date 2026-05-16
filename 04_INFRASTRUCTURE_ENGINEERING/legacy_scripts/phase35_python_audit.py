import os, json, csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'python_code_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

findings = []

target_dirs = [
    ROOT / 'MANIPULANTE',
    ROOT / 'BOT_V2_DAYTIME_LAB' / 'src'
]

risk_keywords = [
    r"order_send",
    r"MT5\.order_send",
    r"allow_live\s*=\s*True",
    r"auto_order_execution\s*=\s*True",
    r"C:\\Users\\",
    r"C:\\BOT DE TRADING",
    r"password",
    r"login",
    r"token"
]

for tdir in target_dirs:
    if not tdir.exists(): continue
    for p in tdir.rglob('*.py'):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
        except: continue
        
        for i, line in enumerate(lines):
            for kw in risk_keywords:
                if re.search(kw, line, re.IGNORECASE):
                    # Ignorar falsos positivos como comentarios o configuraciones permitidas en tests
                    if "Path(__file__)" in line: continue # Pathlib safe
                    
                    findings.append({
                        "file": str(p.relative_to(ROOT)),
                        "line": i + 1,
                        "match": line.strip()[:100],
                        "keyword": kw,
                        "severity": "BLOCKER" if "order_send" in kw.lower() or "True" in kw else "WARNING"
                    })

verdict = "BLOCKER" if any(f['severity'] == 'BLOCKER' for f in findings) else "PASS"
res = {"verdict": verdict, "findings": findings}

with open(AUDIT_DIR / 'phase35_python_code_audit.json', 'w') as f:
    json.dump(res, f, indent=2)

with open(AUDIT_DIR / 'phase35_python_findings.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["file", "line", "match", "keyword", "severity"])
    writer.writeheader()
    writer.writerows(findings)

md = [f"# PYTHON CODE AUDIT\nVerdict: {verdict}\n\n| File | Line | Keyword | Severity |", "|---|---|---|---|"]
for f in findings:
    md.append(f"| {f['file']} | {f['line']} | {f['keyword']} | {f['severity']} |")

with open(AUDIT_DIR / 'phase35_python_code_audit.md', 'w') as f:
    f.write('\n'.join(md))

print(json.dumps(res, indent=2))
