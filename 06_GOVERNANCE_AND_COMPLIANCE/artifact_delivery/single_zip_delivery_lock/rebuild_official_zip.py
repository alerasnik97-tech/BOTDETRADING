import os
import zipfile
import hashlib
import json

# Configuración de Archivos
OUTPUT_TMP_ZIP = "000_PARA_CHATGPT_BUILDING.tmp.zip"
OUTPUT_FINAL_ZIP = "000_PARA_CHATGPT.zip"
VERIFICATION_TXT = os.path.join("06_GOVERNANCE_AND_COMPLIANCE", "artifact_delivery", "single_zip_delivery_lock", "FINAL_ROOT_ZIP_RESTORE_VERIFICATION.txt")
HASH_TXT = os.path.join("06_GOVERNANCE_AND_COMPLIANCE", "artifact_delivery", "single_zip_delivery_lock", "current_zip_hash.txt")

INCLUDED_ROOTS = [
    "01_CORE_PRODUCTION",
    "02_INCUBATION_STAGING",
    "06_GOVERNANCE_AND_COMPLIANCE",
    "08_CLOUD_FREE_RUN_LAB",
]

RESEARCH_BASE = os.path.join("03_RESEARCH_LAB", "BOT_V2_DAYTIME_LAB")
RESEARCH_SRC = os.path.join(RESEARCH_BASE, "src")
RESEARCH_REPORTS = os.path.join(RESEARCH_BASE, "reports")

EXCLUDE_DIRS = {
    ".git", "venv", "venv_v37", ".venv", "env", "cache", "caches", "checkpoints",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "backups", "scratch",
    "archive_scratch", "outputs", "tick", "ticks", "raw", "monthly", "forex", "data", "manual_data",
    "_CHATGPT_EXPORT", "temp_zip_extract"
}

EXCLUDE_EXTENSIONS = {
    ".parquet", ".h5", ".feather", ".bin", ".db", ".sqlite", ".zip", ".7z", ".rar",
    ".bundle", ".bundle.lock", ".log", ".tmp", ".temp", ".bak", ".pyc", ".zipbak", ".csvbak"
}

EXCLUDE_FILENAMES = {
    ".env", "kaggle.json", "mt5_local_config.json"
}

def should_exclude_file(filepath, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in EXCLUDE_EXTENSIONS:
        return True
    
    if filename in EXCLUDE_FILENAMES or filename.startswith(".env"):
        return True
        
    # Excluir archivos masivos y resultados activos cambiantes
    if filename.endswith("TRADES.csv") or "PARTIAL" in filename:
        return True
        
    try:
        # Límite de 2MB por archivo individual para asegurar un ZIP ultra liviano
        if os.path.getsize(filepath) > 2 * 1024 * 1024:
            return True
    except Exception:
        return True
        
    return False

def should_exclude_dir(dirname):
    return dirname in EXCLUDE_DIRS

def build_temp_zip():
    print(f"Iniciando empaquetado atómico en {OUTPUT_TMP_ZIP}...")
    files_to_add = []
    
    if os.path.exists(".gitignore"):
        files_to_add.append(".gitignore")
        
    for root_dir in INCLUDED_ROOTS:
        if not os.path.exists(root_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not should_exclude_file(fp, f):
                    files_to_add.append(fp)
                    
    for base_src in [RESEARCH_SRC, RESEARCH_REPORTS]:
        if os.path.exists(base_src):
            for dirpath, dirnames, filenames in os.walk(base_src):
                dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not should_exclude_file(fp, f):
                        files_to_add.append(fp)
                        
    if os.path.exists(RESEARCH_BASE):
        for f in os.listdir(RESEARCH_BASE):
            fp = os.path.join(RESEARCH_BASE, f)
            if os.path.isfile(fp) and f.endswith(".py") and not should_exclude_file(fp, f):
                files_to_add.append(fp)
                
    # Eliminar duplicados preservando orden
    files_to_add = list(dict.fromkeys(files_to_add))
    
    with zipfile.ZipFile(OUTPUT_TMP_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fp in files_to_add:
            zf.write(fp, fp)
            
    print(f"Empaquetado temporal completado. Total archivos pre-compresión: {len(files_to_add)}")

def verify_and_promote():
    print("Verificando integridad del archivo temporal ZIP...")
    if not os.path.exists(OUTPUT_TMP_ZIP):
        print("ERROR: El archivo temporal no fue creado.")
        return False, "ZIP_RESTORE_BLOCKED", {}
        
    with zipfile.ZipFile(OUTPUT_TMP_ZIP, 'r') as z:
        testzip_result = z.testzip()
        namelist = z.namelist()
        
    total_files = len(namelist)
    size_mb = os.path.getsize(OUTPUT_TMP_ZIP) / (1024 * 1024)
    
    with open(OUTPUT_TMP_ZIP, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    no_internal_zips = all(not name.lower().endswith(".zip") for name in namelist)
    no_raw_data = all(not any(p in name.lower() for p in ["/tick/", "/ticks/", ".parquet", "/raw/"]) for name in namelist)
    no_parquet = all(not name.lower().endswith(".parquet") for name in namelist)
    no_venv = all(not any(p in name.lower() for p in ["venv/", "venv_v37/", ".venv/"]) for name in namelist)
    no_git_cache = all(not any(p in name.lower() for p in [".git/", "__pycache__/", ".pytest_cache/"]) for name in namelist)
    
    contains_governance = any(name.startswith("06_GOVERNANCE_AND_COMPLIANCE/") for name in namelist)
    contains_cloud_lab = any(name.startswith("08_CLOUD_FREE_RUN_LAB/") for name in namelist)
    contains_benchmark = any("benchmark" in name.lower() or "target_objective" in name.lower() for name in namelist)
    contains_incubation = any("incubation" in name.lower() for name in namelist)
    contains_research_reports = any("reports/" in name.lower() for name in namelist)
    
    all_passed = (
        testzip_result is None and no_internal_zips and no_raw_data and no_parquet and
        no_venv and no_git_cache and contains_governance and contains_cloud_lab and
        contains_benchmark and contains_incubation and contains_research_reports
    )
    
    missing_required = []
    if not contains_governance: missing_required.append("governance")
    if not contains_cloud_lab: missing_required.append("cloud_lab")
    if not contains_benchmark: missing_required.append("benchmark")
    if not contains_incubation: missing_required.append("incubation")
    if not contains_research_reports: missing_required.append("research_reports")
    
    status = "ZIP_RESTORE_BLOCKED"
    if all_passed:
        print("Auditoría exitosa. Promocionando temporal a oficial de forma atómica...")
        os.replace(OUTPUT_TMP_ZIP, OUTPUT_FINAL_ZIP)
        status = "ZIP_ROOT_RESTORED"
    else:
        print(f"Fallo en aserciones de auditoría. missing: {missing_required}, testzip: {testzip_result}")
        # No borramos el oficial anterior bajo ninguna circunstancia
        if os.path.exists(OUTPUT_TMP_ZIP):
            os.remove(OUTPUT_TMP_ZIP)
            
    # Conteo final en raíz
    root_zips = [f for f in os.listdir('.') if f.lower().endswith('.zip')]
    temp_leftovers = [f for f in root_zips if f != OUTPUT_FINAL_ZIP]
    
    # Limpieza final de restos en la raíz
    for leftover in temp_leftovers:
        try:
            os.remove(leftover)
        except Exception:
            pass
            
    root_zips_final = [f for f in os.listdir('.') if f.lower().endswith('.zip')]
    root_zip_count = len(root_zips_final)
    
    final_exists = os.path.exists(OUTPUT_FINAL_ZIP)
    final_size_mb = os.path.getsize(OUTPUT_FINAL_ZIP) / (1024 * 1024) if final_exists else 0
    final_sha256 = sha256 if final_exists and status == "ZIP_ROOT_RESTORED" else "N/A"
    
    # Escribir reportes de verificación
    os.makedirs(os.path.dirname(VERIFICATION_TXT), exist_ok=True)
    
    report_lines = [
        "FINAL ROOT ZIP RESTORE VERIFICATION",
        f"Path: {os.path.abspath(OUTPUT_FINAL_ZIP)}",
        f"Exists: {final_exists}",
        f"Status: {status}",
        f"Total Files: {total_files}",
        f"Size MB: {final_size_mb:.2f}",
        f"SHA256: {final_sha256}",
        f"Testzip Result: {testzip_result}",
        f"Root Zip Count: {root_zip_count}",
        f"Root Zips Visible: {root_zips_final}",
        f"Missing Required Folders: {missing_required}",
        f"No Internal Zips: {no_internal_zips}",
        f"No Raw Data: {no_raw_data}",
        f"No Parquet: {no_parquet}",
        f"No Venv: {no_venv}",
        f"No Git Cache: {no_git_cache}"
    ]
    
    with open(VERIFICATION_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines) + "\n")
        
    if final_exists and final_sha256 != "N/A":
        with open(HASH_TXT, 'w', encoding='utf-8') as f:
            f.write(final_sha256 + "\n")
            
    # Imprimir un JSON final para que el orquestador capture todo
    result_dict = {
        "status": status,
        "exists": final_exists,
        "total_files": total_files,
        "size_mb": round(final_size_mb, 2),
        "sha256": final_sha256,
        "testzip": testzip_result,
        "root_zip_count": root_zip_count,
        "missing_required": missing_required,
        "temp_leftovers_cleared": len(temp_leftovers)
    }
    print("=== JSON_RESULT ===")
    print(json.dumps(result_dict))
    
    return all_passed, status, result_dict

if __name__ == "__main__":
    build_temp_zip()
    verify_and_promote()
