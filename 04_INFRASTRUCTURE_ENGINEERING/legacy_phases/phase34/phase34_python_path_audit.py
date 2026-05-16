import os, json, csv, re
from pathlib import Path

ROOT = Path(r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo')
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase34_canonical_path_sync_audit' / 'python_path_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'src'
ROOT_SCRIPTS = [ROOT / 'zip_builder.py', ROOT / 'git_operations.py', ROOT / 'inventory_check.py', ROOT / 'preflight_check.py', ROOT / 'validation_check.py', ROOT / 'phase34_preflight.py', ROOT / 'phase34_path_audit.py']

TARGET_SCRIPTS = [
    'canonical_zip_identity_proof.py',
    'cleanup_temp_zips.py',
    'phase31_',
    'phase32_',
    'phase33_',
    'phase34_',
    'run_canonical_',
    'run_phase3'
]

findings = []

def process_file(p):
    if not p.is_file() or p.suffix != '.py':
        return
    name = p.name.lower()
    
    is_target = p in ROOT_SCRIPTS
    if not is_target:
        for t in TARGET_SCRIPTS:
            if t in name:
                is_target = True
                break
                
    if not is_target:
        return
        
    try:
        with open(p, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return
        
    file_modified = False
    
    for i, line in enumerate(lines):
        # We search for "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
        # and replace with: Path(__file__).resolve().parent...
        # But replacing automatically in code can break things.
        # Let's just do a simple replacement for the absolute paths.
        
        # We just look for C:\...
        if 'C:\\' in line and 'Users\\alera' in line:
            findings.append({
                'file': str(p.relative_to(ROOT)),
                'line_num': i + 1,
                'content': line.strip(),
                'issue': 'Hardcoded absolute path C:\\Users\\alera...'
            })
            
            # Simple heuristic replacement for ROOT assignment
            if 'ROOT = Path(' in line or 'ROOT = Path(r' in line:
                # Calculate relative distance from this file to ROOT
                rel = p.relative_to(ROOT)
                parts = len(rel.parts) - 1
                parents = ".parent" * parts
                lines[i] = f"ROOT = Path(__file__).resolve(){parents}\n"
                file_modified = True
                
        elif "C:\\BOT DE TRADING" in line:
             findings.append({
                'file': str(p.relative_to(ROOT)),
                'line_num': i + 1,
                'content': line.strip(),
                'issue': 'Hardcoded absolute path C:\\BOT DE TRADING'
            })

    if file_modified:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Error writing {p}: {e}")

if SRC_DIR.exists():
    for p in SRC_DIR.rglob('*.py'):
        process_file(p)
        
for p in ROOT_SCRIPTS:
    process_file(p)

with open(AUDIT_DIR / 'phase34_python_path_audit.json', 'w') as f:
    json.dump(findings, f, indent=2)

with open(AUDIT_DIR / 'phase34_python_path_findings.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'line_num', 'content', 'issue'])
    writer.writeheader()
    writer.writerows(findings)

md = ["# PYTHON PATH AUDIT FINDINGS\n", "| File | Line | Issue |", "|---|---|---|"]
for item in findings:
    md.append(f"| {item['file']} | {item['line_num']} | {item['issue']} |")

with open(AUDIT_DIR / 'phase34_python_path_audit.md', 'w') as f:
    f.write('\n'.join(md))

print(f"Python path audit completed. Found {len(findings)} issues.")
