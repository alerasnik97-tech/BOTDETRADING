import os, subprocess, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
GIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'final_project_structure_manipulante' / 'git'
GIT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 14: Git Safety Check
before_status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip()
before_branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
before_remote = subprocess.check_output(['git', 'remote', '-v'], cwd=ROOT, text=True).strip()
before_diff = subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True).strip()

safe_to_commit = True
banned = ['.env', '.pkl', 'secret', 'password', 'token', 'credential', '.zipbak']
for line in before_status.splitlines():
    if not line: continue
    path_str = line[3:]
    name = Path(path_str).name.lower()
    if any(b in name for b in banned):
        safe_to_commit = False

zips = [z for z in ROOT.rglob('*.zip') if not z.name.endswith('.zipbak')]
if len(zips) > 1:
    safe_to_commit = False

git_safety_res = {
    'branch': before_branch,
    'remote_exists': 'origin' in before_remote,
    'safe_to_commit': safe_to_commit,
    'zips_count': len(zips)
}

with open(GIT_DIR / 'final_structure_git_status_before_commit.json', 'w') as f:
    json.dump(git_safety_res, f, indent=2)

with open(GIT_DIR / 'final_structure_git_status_before_commit.md', 'w') as f:
    f.write(f"# GIT SAFETY CHECK\nSafe to commit: {safe_to_commit}\nBranch: {before_branch}\n\n```text\n{before_status}\n```")

# Phase 15: Commit & Push
report = {
    'commit_realizado': False,
    'commit_hash': None,
    'push_realizado': False,
    'remote': before_remote,
    'branch': before_branch,
    'files_changed_summary': '',
    'force_push_usado': False,
    'motivo': ''
}

if safe_to_commit:
    to_add = [
        'MANIPULANTE',
        'ESTRATEGIAS',
        'ABRIR_MANIPULANTE_AQUI.txt',
        'ESTRUCTURA_DEL_PROYECTO.md',
        '00_READ_THIS_FIRST.md',
        '01_CURRENT_PROJECT_STATUS.md',
        '01_CURRENT_PROJECT_STATUS.json',
        '02_STRATEGY_AUTHORITY_MAP.md',
        '02_STRATEGY_AUTHORITY_MAP.json',
        'BOT_V2_DAYTIME_LAB/status.json',
        'BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md',
        'ZIP_CONTENTS_MANIFEST.md',
        '000_PARA_CHATGPT.zip',
        'BOT_V2_DAYTIME_LAB/reports/PHASE32E_GLOBAL_WEEKEND_HARD_CLOSE_POLICY_REPORT.md',
        'BOT_V2_DAYTIME_LAB/reports/PHASE32E_GLOBAL_WEEKEND_HARD_CLOSE_POLICY_REPORT.json',
        'BOT_V2_DAYTIME_LAB/src/phase32e_global_weekend_policy_validator.py',
        'BOT_V2_DAYTIME_LAB/outputs/final_project_structure_manipulante/'
    ]
    
    for item in to_add:
        subprocess.run(['git', 'add', item], cwd=ROOT)
    
    commit_msg = "Organize Manipulante authority package and strategy archive"
    res = subprocess.run(['git', 'commit', '-m', commit_msg], cwd=ROOT, capture_output=True, text=True)
    
    if res.returncode == 0:
        report['commit_realizado'] = True
        report['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['files_changed_summary'] = res.stdout
        
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

with open(GIT_DIR / 'final_structure_git_commit_push_report.json', 'w') as f:
    json.dump(report, f, indent=2)

with open(GIT_DIR / 'final_structure_git_commit_push_report.md', 'w') as f:
    md = [
        "# GIT COMMIT PUSH REPORT",
        f"- Commit: {report['commit_realizado']}",
        f"- Hash: {report['commit_hash']}",
        f"- Push: {report['push_realizado']}",
        f"- Motivo: {report['motivo']}"
    ]
    f.write('\n'.join(md))

print(json.dumps(report, indent=2))
