import os, json, sys, subprocess, hashlib, zipfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
PREFLIGHT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'preflight'
PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)

res = {}
res['timestamp'] = datetime.utcnow().isoformat() + 'Z'
res['cwd'] = os.getcwd()
res['root_official'] = str(ROOT)

try:
    res['branch'] = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    res['remote'] = subprocess.check_output(['git', 'remote', '-v'], cwd=ROOT, text=True).strip()
    res['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip()
    res['git_diff'] = subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True).strip()
except Exception as e:
    res['git_error'] = str(e)

res['manipulante_exists'] = (ROOT / 'MANIPULANTE').exists()
res['estrategias_exists'] = (ROOT / 'ESTRATEGIAS').exists()
res['lab_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB').exists()
res['canonical_zip_exists'] = (ROOT / '000_PARA_CHATGPT.zip').exists()

zips = [z for z in ROOT.rglob('*.zip') if not z.name.endswith(('.zipbak', '.building'))]
res['zip_count'] = len(zips)

if res['canonical_zip_exists']:
    p = ROOT / '000_PARA_CHATGPT.zip'
    res['zip_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        with zipfile.ZipFile(p, 'r') as zf:
            res['testzip'] = zf.testzip()
    except Exception as e:
        res['testzip'] = str(e)

res['manipulante_config_exists'] = (ROOT / 'MANIPULANTE' / '01_ESTRATEGIA_AUTORIDAD' / 'manipulante_config.json').exists()
res['phase25_config_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB' / 'configs' / 'phase25_forward_demo_candidate_config.json').exists()
res['phase25_hash_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB' / 'configs' / 'phase25_forward_demo_candidate_config_hash.txt').exists()
res['phase32e_report_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB' / 'reports' / 'PHASE32E_GLOBAL_WEEKEND_HARD_CLOSE_POLICY_REPORT.md').exists()
res['project_restructure_report_exists'] = (ROOT / 'BOT_V2_DAYTIME_LAB' / 'reports' / 'PROJECT_RESTRUCTURE_MANIPULANTE_FINAL_REPORT.md').exists()
res['mt5_launcher_exists'] = (ROOT / 'MANIPULANTE' / '03_MT5_DEMO_LAUNCHER' / 'ABRIR_MANIPULANTE_DEMO.bat').exists()
res['templates_exists'] = (ROOT / 'MANIPULANTE' / '06_TEMPLATES').exists()
res['checklists_exists'] = (ROOT / 'MANIPULANTE' / '08_CHECKLISTS').exists()

res['no_mt5_real'] = True
res['no_orders'] = True
res['no_autotrading'] = True
res['no_broker_real'] = True

blockers = []
if res.get('branch') != 'main': blockers.append("Not on main branch")
if res['zip_count'] > 1: blockers.append(f"Multiple ZIPs found: {res['zip_count']}")
if not res['manipulante_exists']: blockers.append("MANIPULANTE missing")
if not res['manipulante_config_exists']: blockers.append("manipulante_config.json missing")
if not res['phase25_config_exists'] or not res['phase25_hash_exists']: blockers.append("Phase25 config/hash missing")
if not res['phase32e_report_exists']: blockers.append("Phase32E report missing")
if '.env' in res['git_status'] or 'secret' in res['git_status'].lower(): blockers.append("Secrets visible in git status")

res['blockers'] = blockers

with open(PREFLIGHT_DIR / 'phase35_preflight.json', 'w') as f:
    json.dump(res, f, indent=2)

md = [
    "# PREFLIGHT - PHASE35",
    f"Timestamp: {res['timestamp']}",
    f"Branch: {res.get('branch', 'ERROR')}",
    f"Blockers: {len(blockers)}"
]
for b in blockers:
    md.append(f"- {b}")

with open(PREFLIGHT_DIR / 'phase35_preflight.md', 'w') as f:
    f.write('\n'.join(md))

print(json.dumps(res, indent=2))
if blockers:
    sys.exit(1)
