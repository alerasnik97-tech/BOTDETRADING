import zipfile, hashlib, os, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / '000_PARA_CHATGPT.zip'
BUILD_PATH = ROOT / '000_PARA_CHATGPT.phase32e_building'
VAL_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'final_project_structure_manipulante' / 'zip_validation'
VAL_DIR.mkdir(parents=True, exist_ok=True)

banned_extensions = {'.zip', '.zipbak', '.building', '.pkl', '.parquet', '.bi5', '.db', '.sqlite', '.dll', '.exe'}
banned_names = {'.env', 'mt5_local_config.json', '.gitignore'}
banned_tokens = ['secret', 'password', 'token', 'credential', 'apikey', 'api_key']

# Banned sub-dirs inside BOT_V2_DAYTIME_LAB
lab_banned_dirs = {'data', '__pycache__', 'scratch', 'data_certification', 'news_fortress', '.venv', '.git'}
root_banned_dirs = {'.git', '.venv', '.venv_fixed', '__pycache__', 'data', 'scratch',
              'legacy_archive_2026', 'quarantine', 'secrets', '.mplconfig',
              '.pkg', '.tmp', '.tmp.driveupload', '.vendor_duka', '.vendor_duka2',
              '.vscode', 'ARCHIVE_SUPERSEDED', 'audits', 'bls_html_samples',
              'data_candidates_2022_2025', 'data_free_2020', 'data_free_bootstrap',
              'data_free_bootstrap_multi', 'data_free_full', 'data_free_retry_test',
              'data_intake_2015_2019', 'data_intake_2020_2026_bidask', 'data_precision',
              'data_precision_raw', 'data_usdjpy_2016_2019', 'data_usdjpy_2016_2021',
              'data_usdjpy_2022_2025', 'docs', 'ecb_stage2_checkpoints',
              'external_scbi_research_harness', 'htf_ny_window_scbi_stage2_checkpoints',
              'institutional_research_candidate_lab', 'legacy', 'manual_trade_chartpacks',
              'micro_pilot_gate', 'micro_pilot_protocol', 'monitoring',
              'mt5_demo_executor_lab', 'mt5_deployment_audit', 'next_hypothesis_discovery_checkpoints',
              'ops_external', 'real_htf_filter_ab_checkpoints', 'real_readiness_gate',
              'research_lab', 'research_scripts', 'results',
              'results_REHEARSAL', 'scbi_2020_2025_durability_checkpoints',
              'scbi_full_campaign_checkpoints', 'scbi_global_validation_checkpoints',
              'scripts', 'shadow_line_lab', 'tests_external', 'VPS_READINESS',
              'STRATEGIES', 'DATA MANUAL', ' legacy_archive_2026',
              'data_certification', 'news_fortress'}

def should_include(path):
    if not path.is_file():
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in banned_extensions:
        return False
    if name in banned_names or any(tok in name for tok in banned_tokens):
        return False

    rel = path.relative_to(ROOT)
    parts = list(rel.parts)

    if len(parts) == 1:
        return name in {'00_read_this_first.md', '01_current_project_status.md',
                        '01_current_project_status.json', '02_strategy_authority_map.md',
                        '02_strategy_authority_map.json', 'zip_contents_manifest.md',
                        'abrir_manipulante_aqui.txt', 'estructura_del_proyecto.md'}

    top = parts[0]

    if top in ['MANIPULANTE', 'ESTRATEGIAS']:
        return suffix in {'.md', '.json', '.csv', '.txt', '.bat', '.ps1'}

    if top == 'BOT_V2_DAYTIME_LAB':
        for part in parts[1:]:
            if part in lab_banned_dirs:
                return False
        
        rel_s = str(rel).replace('\\\\', '/')
        if len(parts) == 2:
            return name in {'status.json', 'zip_contents_manifest.md', 'zip_upload_identity_marker.md'}
        
        subdir = parts[1]
        if subdir in ('reports', 'configs', 'docs', 'templates'):
            return suffix in {'.md', '.json', '.txt', '.csv'}
        if subdir == 'outputs':
            if 'zip_validation' in rel_s:
                return False
            return suffix in {'.md', '.json', '.csv', '.txt'}
        if subdir == 'src':
            if suffix != '.py':
                return False
            include_patterns = ['phase18_h1_fractal_sweep', 'phase18_first_3m_choch',
                               'phase26', 'phase27', 'phase28', 'phase29', 'phase30',
                               'phase31', 'phase32', 'phase37', 'run_phase25', 'run_phase24',
                               'canonical_zip', 'rebuild', 'debug', 'cleanup']
            return any(p in name for p in include_patterns)
        return False

    if top in root_banned_dirs:
        return False

    return False

files = []
for p in ROOT.rglob('*'):
    if should_include(p):
        files.append(p)

files.sort(key=lambda p: str(p.relative_to(ROOT)).replace('\\\\', '/'))

if BUILD_PATH.exists():
    BUILD_PATH.unlink()

with zipfile.ZipFile(BUILD_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        arcname = str(path.relative_to(ROOT)).replace('\\\\', '/')
        zf.write(path, arcname)

with zipfile.ZipFile(BUILD_PATH, 'r') as zf:
    test = zf.testzip()
    names = zf.namelist()
    heavy = [n for n in names if zf.getinfo(n).file_size > 2 * 1024 * 1024]
    secrets_found = [n for n in names if any(tok in n.lower() for tok in ['.env', 'secret', 'password', 'token', 'credential'])]
    internal_zips = [n for n in names if n.lower().endswith(('.zip', '.zipbak'))]

os.replace(str(BUILD_PATH), str(ZIP_PATH))
sha = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
size = ZIP_PATH.stat().st_size

res = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'zip_path': str(ZIP_PATH),
    'size': size,
    'entries': len(names),
    'sha256': sha,
    'testzip': test,
    'has_manipulante': any(n.startswith('MANIPULANTE/') for n in names),
    'has_estrategias': any(n.startswith('ESTRATEGIAS/') for n in names),
    'has_abrir_txt': 'ABRIR_MANIPULANTE_AQUI.txt' in names,
    'has_manipulante_config': 'MANIPULANTE/01_ESTRATEGIA_AUTORIDAD/manipulante_config.json' in names,
    'has_estrategias_index': 'ESTRATEGIAS/00_LEER_PRIMERO/ESTRATEGIAS_INDEX.md' in names,
    'has_phase25_config': 'BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt' in names,
    'no_heavy_data': len(heavy) == 0,
    'no_secrets': len(secrets_found) == 0,
    'single_zip_live': len([z for z in ROOT.rglob('*.zip') if not z.name.endswith('.zipbak')]) == 1
}

with open(VAL_DIR / 'final_structure_zip_validation.json', 'w') as f:
    json.dump(res, f, indent=2)

md = ["# FINAL STRUCTURE ZIP VALIDATION\n"]
for k, v in res.items():
    md.append(f"- **{k}**: {v}")
with open(VAL_DIR / 'final_structure_zip_validation.md', 'w') as f:
    f.write('\n'.join(md))

with open(VAL_DIR / 'final_structure_zip_entries.txt', 'w') as f:
    f.write('\n'.join(names))

print(json.dumps(res, indent=2))
