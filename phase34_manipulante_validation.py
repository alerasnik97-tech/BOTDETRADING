import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase34_canonical_path_sync_audit' / 'manipulante_validation'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

MANIPULANTE_DIR = ROOT / 'MANIPULANTE'

requirements = {
    'README': MANIPULANTE_DIR / '00_LEER_PRIMERO' / 'README_MANIPULANTE.md',
    'strategy_card': MANIPULANTE_DIR / '01_ESTRATEGIA_AUTORIDAD' / 'MANIPULANTE_STRATEGY_CARD.md',
    'manipulante_config': MANIPULANTE_DIR / '01_ESTRATEGIA_AUTORIDAD' / 'manipulante_config.json',
    'hard_close_policy': MANIPULANTE_DIR / '01_ESTRATEGIA_AUTORIDAD' / 'MANIPULANTE_GLOBAL_WEEKEND_POLICY.md',
    'risk_policy': MANIPULANTE_DIR / '04_OPERACION_DIARIA' / 'MANIPULANTE_RISK_POLICY.md',
    'kill_switch': MANIPULANTE_DIR / '04_OPERACION_DIARIA' / 'MANIPULANTE_KILL_SWITCH.md',
    'pre_trade_checklist': MANIPULANTE_DIR / '04_OPERACION_DIARIA' / 'MANIPULANTE_PRE_TRADE_CHECKLIST.md',
    'post_trade_checklist': MANIPULANTE_DIR / '04_OPERACION_DIARIA' / 'MANIPULANTE_POST_TRADE_CHECKLIST.md',
    'mt5_launcher_demo_bat': MANIPULANTE_DIR / '03_MT5_DEMO_LAUNCHER' / 'ABRIR_MANIPULANTE_DEMO.bat',
    'templates_dir': MANIPULANTE_DIR / '06_TEMPLATES',
    'compliance_docs_dir': MANIPULANTE_DIR / '09_COMPLIANCE'
}

results = []
all_ok = True

for req_name, req_path in requirements.items():
    exists = req_path.exists()
    results.append({
        'requirement': req_name,
        'path': str(req_path.relative_to(ROOT)).replace('\\', '/'),
        'exists': exists
    })
    if not exists:
        all_ok = False

# Read config to check auto_trading
try:
    with open(requirements['manipulante_config'], 'r') as f:
        config = json.load(f)
    
    no_auto = config.get('auto_order_execution') is False
    no_live = config.get('live_trading_allowed') is False
except Exception:
    no_auto = False
    no_live = False
    all_ok = False

results.append({'requirement': 'no_auto_trading', 'path': 'manipulante_config.json', 'exists': no_auto})
results.append({'requirement': 'no_allow_live', 'path': 'manipulante_config.json', 'exists': no_live})

if not no_auto or not no_live:
    all_ok = False

res = {
    'all_ok': all_ok,
    'config_ok': True if requirements['manipulante_config'].exists() else False,
    'hard_close_ok': True if requirements['hard_close_policy'].exists() else False,
    'risk_policy_ok': True if requirements['risk_policy'].exists() else False,
    'mt5_launcher_safe_ok': True if requirements['mt5_launcher_demo_bat'].exists() and no_auto and no_live else False,
    'checks': results
}

with open(AUDIT_DIR / 'phase34_manipulante_validation.json', 'w') as f:
    json.dump(res, f, indent=2)

with open(AUDIT_DIR / 'phase34_manipulante_required_files.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['requirement', 'path', 'exists'])
    writer.writeheader()
    writer.writerows(results)

md = ["# MANIPULANTE VALIDATION\n", f"Overall OK: {all_ok}\n", "| Requirement | Path | Exists |", "|---|---|---|"]
for item in results:
    md.append(f"| {item['requirement']} | {item['path']} | {item['exists']} |")

with open(AUDIT_DIR / 'phase34_manipulante_validation.md', 'w') as f:
    f.write('\n'.join(md))

print(json.dumps(res, indent=2))
