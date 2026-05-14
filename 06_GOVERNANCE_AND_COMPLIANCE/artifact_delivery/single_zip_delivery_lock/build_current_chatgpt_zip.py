import os
import sys
import zipfile
import hashlib
import shutil
from datetime import datetime

def should_exclude(filepath, filename, relpath):
    # Excluir extensiones prohibidas
    ext = os.path.splitext(filename)[1].lower()
    EXCLUDE_EXTS = {
        ".parquet", ".feather", ".h5", ".db", ".sqlite", ".zip", ".7z", ".rar", 
        ".exe", ".dll", ".pyc", ".log", ".tmp", ".temp", ".bak", ".key", ".pem"
    }
    if ext in EXCLUDE_EXTS:
        return True
        
    # Excluir archivos exactos o patrones de secretos
    EXCLUDE_FILES = {".env", "kaggle.json", ".netrc"}
    if filename in EXCLUDE_FILES or filename.startswith(".env."):
        return True
        
    # Excluir partes de la ruta asociadas a cachés, venv, o datos crudos pesados
    parts = set(relpath.replace('\\', '/').split('/'))
    EXCLUDE_PARTS = {
        ".git", "venv", "venv_v37", ".venv", "__pycache__", ".pytest_cache", 
        ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", "backups", "scratch", 
        "archive_scratch", "outputs", "temp_zip_extract", "_CHATGPT_EXPORT",
        "tick", "ticks", "raw"
    }
    if parts.intersection(EXCLUDE_PARTS):
        return True
        
    # Límite de tamaño por archivo individual (50 MB max para evitar desbordamientos)
    try:
        if os.path.getsize(filepath) > 50 * 1024 * 1024:
            return True
    except Exception:
        return True
        
    return False

def main():
    # 1. Detectar raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    os.chdir(project_root)
    print(f"Raíz del proyecto detectada: {project_root}")
    
    official_zip = "000_PARA_CHATGPT.zip"
    temp_zip = "000_PARA_CHATGPT_BUILDING.tmp.zip"
    upload_folder = r"C:\Users\alera\Desktop\UPLOAD_CHATGPT_ACTUAL"
    report_file = os.path.join("06_GOVERNANCE_AND_COMPLIANCE", "artifact_delivery", "single_zip_delivery_lock", "FINAL_CURRENT_CHATGPT_ZIP_VERIFICATION.txt")
    
    # Limpiar cualquier ZIP temporal previo si quedó
    if os.path.exists(temp_zip):
        os.remove(temp_zip)
        
    # 2. Recolectar archivos a incluir
    files_to_include = []
    
    # Incluir .gitignore si existe en la raíz
    if os.path.exists(".gitignore"):
        files_to_include.append(".gitignore")
        
    # Carpetas raíz a incluir completamente (respetando exclusiones)
    ROOT_DIRS = [
        "01_CORE_PRODUCTION",
        "02_INCUBATION_STAGING",
        "04_INFRASTRUCTURE_ENGINEERING",
        "06_GOVERNANCE_AND_COMPLIANCE",
        "08_CLOUD_FREE_RUN_LAB"
    ]
    
    for rdir in ROOT_DIRS:
        if os.path.exists(rdir):
            for root, dirs, files in os.walk(rdir):
                # Filtrar directorios in-situ para no entrar en cachés/git
                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", "venv_v37", ".venv", ".pytest_cache"}]
                for f in files:
                    fpath = os.path.join(root, f)
                    relpath = os.path.relpath(fpath, project_root)
                    if not should_exclude(fpath, f, relpath):
                        files_to_include.append(relpath)
                        
    # R1 Research Lab específico: incluir src y reports, y archivos en la base de la carpeta
    research_base = os.path.join("03_RESEARCH_LAB", "BOT_V2_DAYTIME_LAB")
    if os.path.exists(research_base):
        # Archivos sueltos en la base de research (ej. scripts runners)
        for f in os.listdir(research_base):
            fpath = os.path.join(research_base, f)
            if os.path.isfile(fpath):
                relpath = os.path.relpath(fpath, project_root)
                if not should_exclude(fpath, f, relpath):
                    files_to_include.append(relpath)
                    
        # Subcarpetas src y reports
        for subdir in ["src", "reports"]:
            subpath = os.path.join(research_base, subdir)
            if os.path.exists(subpath):
                for root, dirs, files in os.walk(subpath):
                    dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", "venv_v37", ".venv", ".pytest_cache"}]
                    for f in files:
                        fpath = os.path.join(root, f)
                        relpath = os.path.relpath(fpath, project_root)
                        if not should_exclude(fpath, f, relpath):
                            files_to_include.append(relpath)
                            
    # Eliminar duplicados manteniendo orden
    files_to_include = list(dict.fromkeys(files_to_include))
    print(f"Total de archivos seleccionados para el ZIP: {len(files_to_include)}")
    
    # 3. Crear ZIP temporal
    print(f"Creando ZIP temporal: {temp_zip}...")
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for relpath in files_to_include:
            zf.write(relpath, relpath)
            
    # 4. Verificar ZIP temporal
    print("Verificando el ZIP temporal...")
    with zipfile.ZipFile(temp_zip, 'r') as zf:
        testzip_res = zf.testzip()
        namelist = zf.namelist()
        
    total_files = len(namelist)
    size_mb = os.path.getsize(temp_zip) / (1024 * 1024)
    
    with open(temp_zip, 'rb') as f:
        temp_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    # Verificaciones de contenido crítico
    nl_str = "/".join(namelist).replace('\\', '/')
    contains_v40 = any("v40_r1_absorption_mean_reversion" in n for n in namelist)
    contains_v41 = any("v41_r1_expansion_sweep" in n for n in namelist)
    contains_v42 = any("v42_r1_confirmation_gauntlet" in n for n in namelist)
    contains_v45 = any("v45_r1_evidence_authenticity_audit" in n for n in namelist)
    contains_v47 = any("v47_r1_real_execution_proof_gate" in n for n in namelist)
    contains_v48 = any("v48_r1_real_factory_batched_run" in n for n in namelist)
    contains_v49 = any("v49_r1_real_factory_expansion_batch3" in n for n in namelist)
    contains_acceptance_gate = any("acceptance_gate" in n for n in namelist)
    contains_v49_decision = any("V49_ACCEPTANCE_DECISION" in n for n in namelist)
    
    # Verificaciones de seguridad
    no_raw_data = all(not any(p in n.lower() for p in ["/tick/", "/ticks/", "/raw/"]) for n in namelist)
    no_parquet = all(not n.lower().endswith(".parquet") for n in namelist)
    no_venv = all(not any(p in n.lower() for p in ["venv/", "venv_v37/", ".venv/"]) for n in namelist)
    no_git_cache = all(not any(p in n.lower() for p in [".git/", "__pycache__/", ".pytest_cache/"]) for n in namelist)
    no_internal_zips = all(not n.lower().endswith(".zip") for n in namelist)
    
    # Refinamos no_secrets para evitar falsos positivos con reportes de auditoría que tengan la palabra 'secret'
    no_secrets = all(not any(n.lower().endswith(s) or f"/{s}" in n.lower() or n.lower().startswith(s) for s in [".env", "kaggle.json", ".pem", ".key"]) for n in namelist)
    
    print(f"DEBUG checks: testzip_res={testzip_res}, no_raw_data={no_raw_data}, no_parquet={no_parquet}, no_venv={no_venv}, no_git_cache={no_git_cache}, no_internal_zips={no_internal_zips}, no_secrets={no_secrets}")
    
    missing_critical = []
    if not contains_v40: missing_critical.append("v40")
    if not contains_v41: missing_critical.append("v41")
    if not contains_v42: missing_critical.append("v42")
    if not contains_v47: missing_critical.append("v47")
    if not contains_v48: missing_critical.append("v48")
    if not contains_v49: missing_critical.append("v49")
    
    is_valid = (
        testzip_res is None and no_raw_data and no_parquet and no_venv and 
        no_git_cache and no_internal_zips and no_secrets
    )
    
    if not is_valid:
        print("ERROR: La verificación del ZIP temporal ha fallado. Bloqueando el proceso.")
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        sys.exit(1)
        
    # 5. Reemplazar el ZIP oficial de forma atómica y borrar otros ZIPs en la raíz
    print("Verificación exitosa. Promocionando a ZIP oficial único...")
    if os.path.exists(official_zip):
        os.remove(official_zip)
    os.rename(temp_zip, official_zip)
    
    # Borrar cualquier otro archivo .zip en la raíz
    for item in os.listdir(project_root):
        if item.lower().endswith(".zip") and item != official_zip:
            try:
                os.remove(os.path.join(project_root, item))
                print(f"Eliminado ZIP extra en la raíz: {item}")
            except Exception as e:
                print(f"No se pudo eliminar {item}: {e}")
                
    root_zip_count = len([x for x in os.listdir(project_root) if x.lower().endswith(".zip")])
    
    # 6. Crear copia externa limpia en UPLOAD_CHATGPT_ACTUAL
    print(f"Limpiando directorio externo de subida: {upload_folder}...")
    os.makedirs(upload_folder, exist_ok=True)
    for item in os.listdir(upload_folder):
        if item.lower().endswith(".zip"):
            try:
                os.remove(os.path.join(upload_folder, item))
                print(f"Eliminado ZIP anterior en upload folder: {item}")
            except Exception as e:
                print(f"No se pudo eliminar {item} en upload folder: {e}")
                
    hash12 = temp_sha256[:12]
    upload_zip_name = f"CHATGPT_UPLOAD_CURRENT_{hash12}.zip"
    upload_zip_path = os.path.join(upload_folder, upload_zip_name)
    
    print(f"Copiando a directorio externo: {upload_zip_path}...")
    shutil.copy2(official_zip, upload_zip_path)
    
    with open(upload_zip_path, 'rb') as f:
        upload_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    upload_zip_count = len([x for x in os.listdir(upload_folder) if x.lower().endswith(".zip")])
    same_hash = (temp_sha256 == upload_sha256)
    
    # 7. Generar reporte final
    timestamp = datetime.now().isoformat()
    report_content = f"""================================================================================
FINAL CURRENT CHATGPT ZIP VERIFICATION REPORT
================================================================================
Build Timestamp        : {timestamp}
Official Path          : {os.path.join(project_root, official_zip)}
Upload Copy Path       : {upload_zip_path}
Official SHA256        : {temp_sha256}
Upload SHA256          : {upload_sha256}
Same Hash              : {same_hash}
Total Files            : {total_files}
Size MB                : {size_mb:.2f} MB
Testzip Result         : {testzip_res}
Root Zip Count         : {root_zip_count}
Upload Zip Count       : {upload_zip_count}
--------------------------------------------------------------------------------
CRITICAL CONTENT PRESENCE:
- contains v40         : {contains_v40}
- contains v41         : {contains_v41}
- contains v42         : {contains_v42}
- contains v45         : {contains_v45}
- contains v47         : {contains_v47}
- contains v48         : {contains_v48}
- contains v49         : {contains_v49}
- contains acceptance_gate : {contains_acceptance_gate}
- contains V49_ACCEPTANCE_DECISION : {contains_v49_decision}
Missing Critical Files : {missing_critical if missing_critical else 'NONE'}
--------------------------------------------------------------------------------
SECURITY & EXCLUSIONS:
- excluded raw data    : {no_raw_data}
- excluded parquet     : {no_parquet}
- excluded venv        : {no_venv}
- excluded git/cache   : {no_git_cache}
- excluded internal zips : {no_internal_zips}
- excluded secrets     : {no_secrets}
================================================================================
"""
    os.makedirs(os.path.dirname(os.path.join(project_root, report_file)), exist_ok=True)
    with open(os.path.join(project_root, report_file), 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Reporte de verificación guardado en: {report_file}")
    
    # Imprimir variables clave para la respuesta de la IA
    print("\n" + "="*40)
    print(f"OUTPUT_STATUS=CURRENT_CHATGPT_ZIP_READY")
    print(f"OUTPUT_TOTAL_FILES={total_files}")
    print(f"OUTPUT_SIZE_MB={size_mb:.2f}")
    print(f"OUTPUT_SHA256={temp_sha256}")
    print(f"OUTPUT_TESTZIP={testzip_res}")
    print(f"OUTPUT_ROOT_ZIP_COUNT={root_zip_count}")
    print(f"OUTPUT_UPLOAD_ZIP_PATH={upload_zip_path}")
    print(f"OUTPUT_UPLOAD_SHA256={upload_sha256}")
    print(f"OUTPUT_SAME_HASH={same_hash}")
    print(f"OUTPUT_UPLOAD_ZIP_COUNT={upload_zip_count}")
    print(f"OUTPUT_CONTAINS_V40={'YES' if contains_v40 else 'NO'}")
    print(f"OUTPUT_CONTAINS_V41={'YES' if contains_v41 else 'NO'}")
    print(f"OUTPUT_CONTAINS_V42={'YES' if contains_v42 else 'NO'}")
    print(f"OUTPUT_CONTAINS_V45={'YES' if contains_v45 else 'NO'}")
    print(f"OUTPUT_CONTAINS_V47={'YES' if contains_v47 else 'NO'}")
    print(f"OUTPUT_CONTAINS_V48={'YES' if contains_v48 else 'NO'}")
    print(f"OUTPUT_CONTAINS_V49={'YES' if contains_v49 else 'NO'}")
    print(f"OUTPUT_CONTAINS_ACCEPTANCE_GATE={'YES' if contains_acceptance_gate else 'NO'}")
    print(f"OUTPUT_CONTAINS_V49_DECISION={'YES' if contains_v49_decision else 'NO'}")
    print("="*40)

if __name__ == "__main__":
    main()
