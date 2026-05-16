import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve()
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase34_canonical_path_sync_audit' / 'path_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
TARGET_EXTS = {'.md', '.json', '.py', '.bat', '.ps1'}
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'data'}

findings = []

for p in ROOT.rglob('*'):
    if not p.is_file() or p.suffix.lower() not in TARGET_EXTS:
        continue
    
    # Check if in skipped dirs
    skip = False
    for part in p.parts:
        if part in SKIP_DIRS:
            skip = True
            break
    if skip:
        continue

    # Read lines
    try:
        with open(p, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        continue
    
    file_modified = False
    for i, line in enumerate(lines):
        # We search for wrong paths
        # "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
        # "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo" (without Bot\)
        upper_line = line.upper()
        if "C:\\BOT DE TRADING" in upper_line or "C:\\USERS\\ALERA\\DESKTOP\\BOT DE TRADING" in upper_line:
            # check if it's already the official path
            if OFFICIAL_PATH.lower() not in line.lower():
                findings.append({
                    'file': str(p.relative_to(ROOT)),
                    'line_num': i + 1,
                    'content': line.strip(),
                    'issue': 'Wrong absolute path'
                })
                # If it's a doc, we can auto-fix it if we want to replace it.
                # but let's just audit first. We can fix it in memory and write back.
                
                # Auto-fix: replace anything looking like the wrong path with OFFICIAL_PATH
                if "C:\\BOT DE TRADING" in line:
                    lines[i] = line.replace("C:\\BOT DE TRADING", OFFICIAL_PATH)
                    file_modified = True
                elif "C:\\Users\\alera\\Desktop\\BOT DE TRADING" in line:
                    lines[i] = line.replace("C:\\Users\\alera\\Desktop\\BOT DE TRADING", OFFICIAL_PATH)
                    file_modified = True

    if file_modified:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception:
            pass

# Save findings
with open(AUDIT_DIR / 'phase34_path_audit.json', 'w') as f:
    json.dump(findings, f, indent=2)

with open(AUDIT_DIR / 'phase34_path_findings.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'line_num', 'content', 'issue'])
    writer.writeheader()
    writer.writerows(findings)

md = ["# PATH AUDIT FINDINGS\n", "| File | Line | Issue |", "|---|---|---|"]
for item in findings:
    md.append(f"| {item['file']} | {item['line_num']} | {item['issue']} |")

with open(AUDIT_DIR / 'phase34_path_audit.md', 'w') as f:
    f.write('\n'.join(md))

print(f"Path audit completed. Found {len(findings)} issues.")
