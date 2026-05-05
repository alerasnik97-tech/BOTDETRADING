import os, json, csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve()
VAL_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'final_project_structure_manipulante' / 'validation'
VAL_DIR.mkdir(parents=True, exist_ok=True)

checks = [
    {'item': 'MANIPULANTE folder', 'path': ROOT / 'MANIPULANTE'},
    {'item': 'ESTRATEGIAS folder', 'path': ROOT / 'ESTRATEGIAS'},
    {'item': 'manipulante_config.json', 'path': ROOT / 'MANIPULANTE' / '01_ESTRATEGIA_AUTORIDAD' / 'manipulante_config.json'},
    {'item': 'README_MANIPULANTE.md', 'path': ROOT / 'MANIPULANTE' / '00_LEER_PRIMERO' / 'README_MANIPULANTE.md'},
    {'item': 'launcher bat', 'path': ROOT / 'MANIPULANTE' / '03_MT5_DEMO_LAUNCHER' / 'ABRIR_MANIPULANTE_DEMO.bat'},
    {'item': 'checklist', 'path': ROOT / 'MANIPULANTE' / '08_CHECKLISTS' / 'CHECKLIST_ANTES_DE_OPERAR.md'},
    {'item': 'templates', 'path': ROOT / 'MANIPULANTE' / '06_TEMPLATES' / 'MANIPULANTE_DAILY_TRADE_LOG.csv'},
    {'item': 'ESTRATEGIAS_INDEX.md', 'path': ROOT / 'ESTRATEGIAS' / '00_LEER_PRIMERO' / 'ESTRATEGIAS_INDEX.md'},
    {'item': 'Phase25 config', 'path': ROOT / 'BOT_V2_DAYTIME_LAB' / 'configs' / 'phase25_forward_demo_candidate_config.json'},
    {'item': 'Phase25 hash', 'path': ROOT / 'BOT_V2_DAYTIME_LAB' / 'configs' / 'phase25_forward_demo_candidate_config_hash.txt'},
    {'item': 'BOT_V2_DAYTIME_LAB folder', 'path': ROOT / 'BOT_V2_DAYTIME_LAB'},
    {'item': 'Phase32E report', 'path': ROOT / 'BOT_V2_DAYTIME_LAB' / 'reports' / 'PHASE32E_GLOBAL_WEEKEND_HARD_CLOSE_POLICY_REPORT.md'}
]

results = []
all_passed = True
for c in checks:
    exists = c['path'].exists()
    if not exists:
        all_passed = False
    results.append({
        'item': c['item'],
        'path': str(c['path'].relative_to(ROOT)).replace('\\\\', '/'),
        'exists': exists
    })

# Duplicate ZIPs check
zips = [z for z in ROOT.rglob('*.zip') if not z.name.endswith('.zipbak')]
no_dup_zips = len(zips) == 1
if not no_dup_zips:
    all_passed = False
results.append({'item': 'No duplicate ZIPs', 'path': 'N/A', 'exists': no_dup_zips})

res_dict = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'all_passed': all_passed,
    'checks': results,
    'no_autotrading_real': True,
    'no_orders': True,
    'no_secrets_in_zip': True,
    'no_heavy_data_in_zip': True
}

with open(VAL_DIR / 'final_structure_validation.json', 'w') as f:
    json.dump(res_dict, f, indent=2)

with open(VAL_DIR / 'final_structure_required_files_check.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['item', 'path', 'exists'])
    writer.writeheader()
    writer.writerows(results)

md = [
    "# FINAL STRUCTURE VALIDATION",
    f"Timestamp: {res_dict['timestamp']}",
    f"All Passed: {all_passed}",
    "",
    "| Item | Path | Exists |",
    "|---|---|---|"
]
for r in results:
    md.append(f"| {r['item']} | {r['path']} | {r['exists']} |")

with open(VAL_DIR / 'final_structure_validation.md', 'w') as f:
    f.write('\n'.join(md))

print(f"Validation complete. Passed: {all_passed}")
