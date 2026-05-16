import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve()
INV_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'final_project_structure_manipulante' / 'inventory'
INV_DIR.mkdir(parents=True, exist_ok=True)

inventory = []

for root, dirs, files in os.walk(ROOT):
    p = Path(root)
    if '.git' in p.parts or '.venv' in p.parts or '__pycache__' in p.parts:
        continue
    for f in files:
        fpath = p / f
        rel = fpath.relative_to(ROOT)
        size = fpath.stat().st_size
        
        # Classification
        classification = "UNKNOWN_REVIEW_REQUIRED"
        name = f.lower()
        parts = rel.parts
        
        if size > 2 * 1024 * 1024 and not f.endswith('.zip'):
            classification = "HEAVY_DATA_DO_NOT_ZIP"
        elif any(t in name for t in ['secret', 'password', 'token', 'credential']) or name == '.env':
            classification = "POSSIBLE_SECRET_REVIEW"
        elif parts[0] in ['Manipulante', 'MANIPULANTE']:
            classification = "MANIPULANTE_REQUIRED"
        elif parts[0] == 'ESTRATEGIAS':
            classification = "ESTRATEGIAS_REQUIRED"
        elif len(parts) == 1 and name in ['00_read_this_first.md', '01_current_project_status.md', '01_current_project_status.json', '02_strategy_authority_map.md', '02_strategy_authority_map.json', 'zip_contents_manifest.md', '000_para_chatgpt.zip']:
            classification = "CORE_KEEP"
        elif 'phase25' in name or 'phase32' in name:
            classification = "CURRENT_AUTHORITY"
        elif 'be05' in name:
            classification = "SHADOW_REFERENCE"
        elif 'phase18' in name:
            classification = "BASELINE_REFERENCE"
        elif 'phase19' in name:
            classification = "ARCHIVED_DO_NOT_USE"
        elif 'no_be' in name:
            classification = "REJECTED_STRATEGY"
        elif 'phase28' in name or 'phase29' in name:
            classification = "EXPERIMENTAL_RESEARCH"
            
        inventory.append({
            'path': str(rel).replace('\\\\', '/'),
            'name': f,
            'size': size,
            'classification': classification
        })

with open(INV_DIR / 'project_inventory_full.json', 'w') as f:
    json.dump(inventory, f, indent=2)

with open(INV_DIR / 'project_inventory_full.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'name', 'size', 'classification'])
    writer.writeheader()
    writer.writerows(inventory)

md = ["# PROJECT INVENTORY FULL\n", "| Path | Classification | Size |", "|---|---|---|"]
for item in sorted(inventory, key=lambda x: x['classification']):
    md.append(f"| {item['path']} | {item['classification']} | {item['size']} |")

with open(INV_DIR / 'project_inventory_full.md', 'w') as f:
    f.write('\n'.join(md))

print(f"Inventory saved: {len(inventory)} items")
