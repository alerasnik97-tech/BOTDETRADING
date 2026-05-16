import os, json, csv, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase34_canonical_path_sync_audit' / 'core_docs_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
CORE_PROTOCOLS_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'docs' / 'CORE_PROTOCOLS'
CORE_PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DOCS = [
    'AGENTS.MD',
    'RISK_PROTOCOL.MD',
    'RESEARCH_OPERATING_SYSTEM.MD',
    'CLOUD_WORKFLOW.MD',
    'AI_AGENT_MANIFESTO.MD'
]

findings = []
doc_map = {doc: [] for doc in TARGET_DOCS}

for p in ROOT.rglob('*.md'):
    name = p.name.upper()
    if name in TARGET_DOCS:
        # Ignore if it's already in the canonical location
        if p.parent == CORE_PROTOCOLS_DIR:
            continue
        # Also ignore if it's in a .git or something
        if '.git' in p.parts:
            continue
            
        doc_map[name].append(p)
        findings.append({
            'doc': name,
            'path': str(p.relative_to(ROOT)),
            'issue': 'Duplicate core document found outside canonical location'
        })

# Consolidate
for doc, paths in doc_map.items():
    if not paths:
        continue
    
    canonical_path = CORE_PROTOCOLS_DIR / doc
    
    # We take the first one (most likely from a root or older docs folder) and copy it to canonical
    # We try to find the largest file to assume it's the most complete
    best_path = max(paths, key=lambda x: x.stat().st_size)
    
    if not canonical_path.exists():
        shutil.copy2(best_path, canonical_path)
        
    # Mark the old ones as legacy
    legacy_text = f"\n\n> **ESTE ARCHIVO ES UN SNAPSHOT HISTÓRICO.**\n> La fuente viva y canónica está en: `BOT_V2_DAYTIME_LAB/docs/CORE_PROTOCOLS/{doc}`\n"
    
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                content = f.read()
            if "ESTE ARCHIVO ES UN SNAPSHOT HISTÓRICO" not in content:
                with open(p, 'a', encoding='utf-8') as f:
                    f.write(legacy_text)
        except Exception:
            pass

with open(AUDIT_DIR / 'phase34_core_docs_audit.json', 'w') as f:
    json.dump(findings, f, indent=2)

with open(AUDIT_DIR / 'phase34_core_docs_duplicates.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['doc', 'path', 'issue'])
    writer.writeheader()
    writer.writerows(findings)

md = ["# CORE DOCS AUDIT FINDINGS\n", "| Doc | Path | Issue |", "|---|---|---|"]
for item in findings:
    md.append(f"| {item['doc']} | {item['path']} | {item['issue']} |")

with open(AUDIT_DIR / 'phase34_core_docs_audit.md', 'w') as f:
    f.write('\n'.join(md))

print(f"Core docs audit completed. Found {len(findings)} duplicates. Canonical sources created/updated.")
