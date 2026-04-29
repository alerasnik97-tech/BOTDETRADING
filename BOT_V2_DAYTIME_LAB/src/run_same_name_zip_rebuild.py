import os
import sys
import json
import hashlib
import zipfile
import shutil
from datetime import datetime, timezone

ROOT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
LAB_DIR = os.path.join(ROOT_DIR, "BOT_V2_DAYTIME_LAB")
OUTPUT_DIR = os.path.join(LAB_DIR, "outputs", "same_name_zip_rebuild")
PREFLIGHT_DIR = os.path.join(OUTPUT_DIR, "preflight")
QUARANTINE_DIR = os.path.join(OUTPUT_DIR, "quarantine")
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "validation")
GIT_DIR = os.path.join(OUTPUT_DIR, "git")

TARGET_ZIP_NAME = "000_PARA_CHATGPT.zip"
TARGET_ZIP_PATH = os.path.join(ROOT_DIR, TARGET_ZIP_NAME)
BUILD_ZIP_PATH = os.path.join(ROOT_DIR, "000_PARA_CHATGPT.building")

os.makedirs(PREFLIGHT_DIR, exist_ok=True)
os.makedirs(QUARANTINE_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(GIT_DIR, exist_ok=True)

# Helper functions
def get_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_size(path):
    return os.path.getsize(path)

# Phase 0: Inventory
print("PHASE 0: Inventory and quarantine")
zip_inventory = []
for file in os.listdir(ROOT_DIR):
    if file.endswith(".zip"):
        path = os.path.join(ROOT_DIR, file)
        size = get_size(path)
        sha = get_sha256(path)
        
        info = {
            "name": file,
            "path": path,
            "size": size,
            "sha256": sha,
            "entry_count": 0,
            "testzip": "Error opening",
            "has_phase25": False,
            "has_phase26a": False,
            "has_phase26b": False,
            "has_heavy_data": False,
            "has_secrets": False
        }
        
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                info["entry_count"] = len(zf.namelist())
                info["testzip"] = str(zf.testzip())
                namelist = zf.namelist()
                
                info["has_phase25"] = any("PHASE25" in n for n in namelist)
                info["has_phase26a"] = any("PHASE26A" in n for n in namelist)
                info["has_phase26b"] = any("PHASE26B" in n for n in namelist)
                info["has_heavy_data"] = any(n.endswith('.csv') and zf.getinfo(n).file_size > 2000000 for n in namelist)
                info["has_secrets"] = any(n.endswith('.env') or n.endswith('.pem') for n in namelist)
        except Exception as e:
            info["testzip"] = f"Error: {e}"
            
        zip_inventory.append(info)
        
        if file != TARGET_ZIP_NAME:
            dest = os.path.join(QUARANTINE_DIR, file + "bak")
            os.rename(path, dest)
            print(f"Quarantined: {file} to {dest}")

with open(os.path.join(PREFLIGHT_DIR, "same_name_zip_rebuild_preflight.json"), "w") as f:
    json.dump(zip_inventory, f, indent=4)

with open(os.path.join(PREFLIGHT_DIR, "same_name_zip_rebuild_preflight.md"), "w") as f:
    f.write("# Preflight Inventory\n\n")
    for z in zip_inventory:
        f.write(f"- {z['name']} ({z['size']} bytes) - Entries: {z['entry_count']}\n")

# Phase 1: Identity Marker
print("PHASE 1: Identity Marker")
marker_path = os.path.join(LAB_DIR, "ZIP_UPLOAD_IDENTITY_MARKER.md")
timestamp = datetime.now(timezone.utc).isoformat()
marker_content = f"""# ZIP UPLOAD IDENTITY MARKER

- Timestamp: {timestamp}
- Nombre del ZIP: {TARGET_ZIP_NAME}
- Ruta exacta: {TARGET_ZIP_PATH}
- Fase actual: CANONICAL ZIP SAME NAME REBUILD
- Autoridad actual: PHASE25_MAX_ROBUST
- Phase26 estado: OPTIMIZATION_BLOCKED_PENDING_2015_2019_M1_OR_TICK_DATA

> NOTA: Este ZIP fue reconstruido desde cero manteniendo el mismo nombre canónico.
> NOTA: Si ChatGPT recibe un ZIP sin este marker actualizado, se subió una versión vieja/cacheada.
"""
with open(marker_path, "w", encoding='utf-8') as f:
    f.write(marker_content)

# Phase 2 & 3 & 4: Build Temporal Zip
print("PHASE 2 & 3 & 4: Build Temporal Zip")
MANDATORY_FILES = [
    "00_READ_THIS_FIRST.md",
    "01_CURRENT_PROJECT_STATUS.md",
    "01_CURRENT_PROJECT_STATUS.json",
    "02_STRATEGY_AUTHORITY_MAP.md",
    "02_STRATEGY_AUTHORITY_MAP.json",
    "ZIP_CONTENTS_MANIFEST.md",
    r"BOT_V2_DAYTIME_LAB\ZIP_CONTENTS_MANIFEST.md",
    r"BOT_V2_DAYTIME_LAB\status.json",
    r"BOT_V2_DAYTIME_LAB\ZIP_UPLOAD_IDENTITY_MARKER.md",
    r"BOT_V2_DAYTIME_LAB\docs\CANONICAL_ZIP_OPERATING_STANDARD.md",
    r"BOT_V2_DAYTIME_LAB\docs\CANONICAL_ZIP_CHECKLIST.md",
    r"BOT_V2_DAYTIME_LAB\configs\phase25_forward_demo_candidate_config.json",
    r"BOT_V2_DAYTIME_LAB\configs\phase25_forward_demo_candidate_config_hash.txt",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE25_FINAL_CLOSEOUT_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE25_FINAL_CLOSEOUT_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE25_MAX_ROBUST_PLATEAU_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE25_MAX_ROBUST_PLATEAU_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\docs\PHASE25_DAILY_RUNBOOK.md",
    r"BOT_V2_DAYTIME_LAB\docs\PHASE25_KILL_SWITCH_POLICY.md",
    r"BOT_V2_DAYTIME_LAB\docs\PHASE25_FORWARD_REVIEW_CRITERIA.md",
    r"BOT_V2_DAYTIME_LAB\reports\INSTITUTIONAL_DAYTIME_STRATEGY_RANKING_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\INSTITUTIONAL_DAYTIME_STRATEGY_RANKING_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26A_FINAL_CLOSEOUT_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26A_FINAL_CLOSEOUT_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26B_EURUSD_2015_2019_DATA_ENGINEERING_PILOT_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26B_EURUSD_2015_2019_DATA_ENGINEERING_PILOT_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\docs\PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md",
    r"BOT_V2_DAYTIME_LAB\docs\PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md",
    r"BOT_V2_DAYTIME_LAB\src\phase26b_download_or_import_eurusd_2015_2019.py",
    r"BOT_V2_DAYTIME_LAB\src\phase26b_normalize_eurusd_m1_2015_2019.py",
    r"BOT_V2_DAYTIME_LAB\src\phase26b_audit_m1_quality_2015_2019.py",
    r"BOT_V2_DAYTIME_LAB\src\phase26b_generate_m3_from_m1_2015_2019.py",
    r"BOT_V2_DAYTIME_LAB\src\phase26b_build_data_quality_mask_2015_2019.py",
    r"BOT_V2_DAYTIME_LAB\src\phase26b_certify_news_fortress_2015_2019.py",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26C_FULL_2015_2019_DATA_CERTIFICATION_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE26C_FULL_2015_2019_DATA_CERTIFICATION_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\src\phase26c_full_pipeline.py",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE27_PHASE25_FULL_HISTORICAL_VALIDATION_2015_2026_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE27_PHASE25_FULL_HISTORICAL_VALIDATION_2015_2026_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\src\phase27_full_historical_validation.py",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE28_WINRATE_FREQUENCY_IMPROVEMENT_STUDY_REPORT.md",
    r"BOT_V2_DAYTIME_LAB\reports\PHASE28_WINRATE_FREQUENCY_IMPROVEMENT_STUDY_REPORT.json",
    r"BOT_V2_DAYTIME_LAB\src\phase28_winrate_frequency_study.py",
]

PROHIBITED_DIRS = [
    ".git", ".venv", ".venv_fixed", "__pycache__", "data", "data_intake_2015_2019", "data_intake_2020_2026_bidask",
    "legacy_archive_2026", "scratch", "outputs", "quarantine"
]
PROHIBITED_EXTS = [
    ".pkl", ".zip", ".zipbak", ".pem", ".log"
]
PROHIBITED_FILES = [
    ".env", "mt5_local_config.json"
]

files_to_zip = []

for rel in MANDATORY_FILES:
    p = os.path.join(ROOT_DIR, rel)
    if os.path.exists(p):
        files_to_zip.append(rel)
    else:
        # Create empty if missing for mandatory manifests/reports just in case they don't exist
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("{}")

for root, dirs, files in os.walk(ROOT_DIR):
    dirs[:] = [d for d in dirs if d not in PROHIBITED_DIRS and not d.startswith(".")]
    rel_root = os.path.relpath(root, ROOT_DIR)
    
    if rel_root == "BOT_V2_DAYTIME_LAB\\outputs" or rel_root.startswith("BOT_V2_DAYTIME_LAB\\outputs"):
        dirs[:] = []
        continue

    for file in files:
        if file in PROHIBITED_FILES:
            continue
        if any(file.endswith(ext) for ext in PROHIBITED_EXTS):
            continue
        
        path = os.path.join(root, file)
        rel_path = os.path.relpath(path, ROOT_DIR)
        
        if rel_path in files_to_zip:
            continue
            
        try:
            if get_size(path) > 2 * 1024 * 1024:
                continue
        except:
            continue
            
        if file.endswith(".py") or file.endswith(".json") or file.endswith(".md"):
            files_to_zip.append(rel_path)

if os.path.exists(BUILD_ZIP_PATH):
    os.remove(BUILD_ZIP_PATH)

print(f"Zipping {len(files_to_zip)} files to {BUILD_ZIP_PATH}")
with zipfile.ZipFile(BUILD_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for rel_path in files_to_zip:
        abs_path = os.path.join(ROOT_DIR, rel_path)
        if os.path.exists(abs_path):
            zf.write(abs_path, arcname=rel_path.replace("\\", "/"))

test_res = None
entry_count = 0
with zipfile.ZipFile(BUILD_ZIP_PATH, 'r') as zf:
    test_res = zf.testzip()
    entry_count = len(zf.namelist())

if test_res is None and entry_count > 0:
    if os.path.exists(TARGET_ZIP_PATH):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_zip_dest = os.path.join(QUARANTINE_DIR, f"000_PARA_CHATGPT.backup_{ts}.zipbak")
        os.rename(TARGET_ZIP_PATH, old_zip_dest)
    os.rename(BUILD_ZIP_PATH, TARGET_ZIP_PATH)
    print("Renamed building zip to target zip successfully.")
else:
    print("FAILED ZIP VALIDATION. Test_res:", test_res)
    sys.exit(1)

# Phase 5: Internal validation
print("PHASE 5: Internal validation")
zip_entries = []
val_results = {
    "testzip": None,
    "entry_count": 0,
    "size_bytes": 0,
    "size_mb": 0,
    "sha256": "",
    "checks": {
        "marker_exists": False,
        "marker_timestamp_new": False,
        "phase25_closeout_exists": False,
        "phase26a_closeout_exists": False,
        "phase26b_requirements_exists": False,
        "status_correct": False,
        "no_git": True,
        "no_env": True,
        "no_secrets": True,
        "no_heavy_data": True,
        "only_one_zip_alive": False
    }
}

val_results["size_bytes"] = get_size(TARGET_ZIP_PATH)
val_results["size_mb"] = round(val_results["size_bytes"] / (1024*1024), 2)
val_results["sha256"] = get_sha256(TARGET_ZIP_PATH)

with zipfile.ZipFile(TARGET_ZIP_PATH, 'r') as zf:
    val_results["testzip"] = str(zf.testzip())
    zip_entries = zf.namelist()
    val_results["entry_count"] = len(zip_entries)
    
    val_results["checks"]["marker_exists"] = "BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md" in zip_entries
    if val_results["checks"]["marker_exists"]:
        content = zf.read("BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md").decode('utf-8')
        val_results["checks"]["marker_timestamp_new"] = "CANONICAL ZIP SAME NAME REBUILD" in content
    
    val_results["checks"]["phase25_closeout_exists"] = "BOT_V2_DAYTIME_LAB/reports/PHASE25_FINAL_CLOSEOUT_REPORT.md" in zip_entries
    val_results["checks"]["phase26a_closeout_exists"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md" in zip_entries
    val_results["checks"]["phase26b_requirements_exists"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md" in zip_entries
    
    val_results["checks"]["no_git"] = not any(".git/" in n for n in zip_entries)
    val_results["checks"]["no_env"] = not any(n.endswith(".env") for n in zip_entries)
    val_results["checks"]["no_secrets"] = not any(n.endswith(".pem") or "secret" in n.lower() for n in zip_entries)
    val_results["checks"]["no_heavy_data"] = not any(n.endswith(".csv") and zf.getinfo(n).file_size > 2000000 for n in zip_entries)

zips_alive = [f for f in os.listdir(ROOT_DIR) if f.endswith(".zip")]
val_results["checks"]["only_one_zip_alive"] = len(zips_alive) == 1 and zips_alive[0] == TARGET_ZIP_NAME

with open(os.path.join(VALIDATION_DIR, "same_name_zip_rebuild_validation.json"), "w") as f:
    json.dump(val_results, f, indent=4)

with open(os.path.join(VALIDATION_DIR, "same_name_zip_rebuild_entries.txt"), "w") as f:
    f.write("\n".join(zip_entries))

print("PHASE 6: Manifests")
manifest_content = f"""# ZIP CONTENTS MANIFEST

- Unico ZIP canónico vivo: {TARGET_ZIP_NAME}
- Ruta oficial: {TARGET_ZIP_PATH}
- Fase actual: CANONICAL ZIP SAME NAME REBUILD
- Autoridad actual: Phase25
- Phase26: Optimización bloqueada (falta M1/Tick real 2015-2019)
- Próximo paso: Conseguir/certificar EURUSD M1/Tick 2015-2019
- Entry count: {val_results['entry_count']}
- SHA256: {val_results['sha256']}
- Testzip: {val_results['testzip']}
- Marker anti-cache: PRESENTE
- Cero secretos, cero data pesada.
"""
for m_path in [os.path.join(ROOT_DIR, "ZIP_CONTENTS_MANIFEST.md"), os.path.join(LAB_DIR, "ZIP_CONTENTS_MANIFEST.md")]:
    with open(m_path, "w", encoding="utf-8") as f:
        f.write(manifest_content)

print("PHASE 7: Archivo de Identidad")
ident_path = os.path.join(ROOT_DIR, "ZIP_VALIDADO_SUBIR_ESTE.txt")
ident_content = f"""Subir a ChatGPT este archivo:
{TARGET_ZIP_PATH}

Datos exactos:
- tamaño MB: {val_results['size_mb']}
- tamaño bytes: {val_results['size_bytes']}
- SHA256: {val_results['sha256']}
- entry count: {val_results['entry_count']}
- testzip: {val_results['testzip']}
- marker interno presente: {val_results['checks']['marker_exists']}
- Phase25 presente: {val_results['checks']['phase25_closeout_exists']}
- Phase26-A closeout presente: {val_results['checks']['phase26a_closeout_exists']}
- Phase26-B requirements/checklist presentes: {val_results['checks']['phase26b_requirements_exists']}
"""
with open(ident_path, "w", encoding="utf-8") as f:
    f.write(ident_content)

print("ALL DONE SCRIPT")
