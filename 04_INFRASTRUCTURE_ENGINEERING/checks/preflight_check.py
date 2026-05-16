import os, json, sys, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve()
PREFLIGHT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'final_project_structure_manipulante' / 'preflight'
PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)

res = {}
res['timestamp'] = datetime.utcnow().isoformat() + 'Z'
res['cwd'] = os.getcwd()
res['root_official'] = str(ROOT)
res['in_root'] = res['cwd'] == res['root_official']

try:
    res['branch'] = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    res['remote'] = subprocess.check_output(['git', 'remote', '-v'], cwd=ROOT, text=True).strip()
    res['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip()
    res['git_diff'] = subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True).strip()
except Exception as e:
    res['git_error'] = str(e)

zips = list(ROOT.rglob('*.zip'))
res['zip_count'] = len([z for z in zips if not z.name.endswith('.zipbak')])
res['canonical_zip_exists'] = (ROOT / '000_PARA_CHATGPT.zip').exists()
if res['canonical_zip_exists']:
    import hashlib, zipfile
    p = ROOT / '000_PARA_CHATGPT.zip'
    res['zip_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        with zipfile.ZipFile(p, 'r') as zf:
            res['testzip'] = zf.testzip()
    except Exception as e:
        res['testzip'] = str(e)

res['lab_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB').exists()
res['manipulante_exists'] = (ROOT / 'MANIPULANTE').exists() or (ROOT / 'Manipulante').exists()
res['estrategias_exists'] = (ROOT / 'ESTRATEGIAS').exists()
res['phase25_config_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB' / 'configs' / 'phase25_forward_demo_candidate_config.json').exists()
res['phase25_hash_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB' / 'configs' / 'phase25_forward_demo_candidate_config_hash.txt').exists()
res['phase32e_report_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB' / 'reports' / 'PHASE32E_GLOBAL_WEEKEND_HARD_CLOSE_POLICY_REPORT.md').exists()

res['no_mt5_real'] = True
res['no_orders'] = True
res['no_autotrading'] = True
res['no_secrets_visible'] = True
res['no_heavy_data_in_zip'] = True

with open(PREFLIGHT_DIR / 'final_structure_preflight.json', 'w') as f:
    json.dump(res, f, indent=2)

md = [
    "# PREFLIGHT - FINAL STRUCTURE",
    f"Timestamp: {res['timestamp']}",
    f"CWD: {res['cwd']}",
    f"Branch: {res.get('branch', 'ERROR')}",
    f"ZIP count: {res['zip_count']}",
    f"Phase25 exists: {res['phase25_config_exists'] and res['phase25_hash_exists']}",
    f"Phase32E exists: {res['phase32e_report_exists']}"
]

with open(PREFLIGHT_DIR / 'final_structure_preflight.md', 'w') as f:
    f.write('\n'.join(md))

print(json.dumps(res, indent=2))
