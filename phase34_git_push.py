import os, subprocess, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Check for banned files in untracked/modified
before_status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip()
safe_to_commit = True
banned = ['.env', '.pkl', 'secret', 'password', 'token', 'credential']
for line in before_status.splitlines():
    if not line: continue
    path_str = line[3:]
    name = Path(path_str).name.lower()
    if any(b in name for b in banned):
        safe_to_commit = False

zips = [z for z in ROOT.rglob('*.zip') if not z.name.endswith('.zipbak')]
if len(zips) > 1:
    safe_to_commit = False

report = {
    'commit_realizado': False,
    'commit_hash': None,
    'push_realizado': False,
    'force_push_usado': False,
    'motivo': ''
}

if safe_to_commit:
    to_add = [
        'MANIPULANTE',
        'ESTRATEGIAS',
        '00_READ_THIS_FIRST.md',
        '01_CURRENT_PROJECT_STATUS.md',
        '01_CURRENT_PROJECT_STATUS.json',
        '02_STRATEGY_AUTHORITY_MAP.md',
        '02_STRATEGY_AUTHORITY_MAP.json',
        'BOT_V2_DAYTIME_LAB/status.json',
        'ZIP_CONTENTS_MANIFEST.md',
        'BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md',
        'ESTRUCTURA_DEL_PROYECTO.md',
        'ABRIR_MANIPULANTE_AQUI.txt',
        'BOT_V2_DAYTIME_LAB/docs/CORE_PROTOCOLS',
        'BOT_V2_DAYTIME_LAB/docs/MANUAL_EXECUTION_BOUNDARY.md',
        'BOT_V2_DAYTIME_LAB/reports',
        'BOT_V2_DAYTIME_LAB/outputs/phase34_canonical_path_sync_audit',
        'BOT_V2_DAYTIME_LAB/src',
        '000_PARA_CHATGPT.zip'
    ]
    
    for item in to_add:
        # Use -f if the item is inside ignored folders
        if 'outputs' in item:
            subprocess.run(['git', 'add', '-f', item], cwd=ROOT)
        else:
            subprocess.run(['git', 'add', item], cwd=ROOT)
    
    commit_msg = "Phase34 canonical path and Manipulante sync audit"
    res = subprocess.run(['git', 'commit', '-m', commit_msg], cwd=ROOT, capture_output=True, text=True)
    
    if res.returncode == 0:
        report['commit_realizado'] = True
        report['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        
        # PUSH
        push_res = subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, capture_output=True, text=True)
        if push_res.returncode == 0:
            report['push_realizado'] = True
            report['motivo'] = 'Exito'
        else:
            report['motivo'] = f'Push failed: {push_res.stderr}'
    else:
        report['motivo'] = f'Commit failed: {res.stdout}'
else:
    report['motivo'] = 'Not safe to commit'

print(json.dumps(report, indent=2))
