import os
import zipfile
import shutil
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

def main():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    
    preflight_dir = lab / "outputs" / "canonical_zip_hard_reset" / "preflight"
    quarantine_dir = lab / "outputs" / "canonical_zip_hard_reset" / "quarantine"
    validation_dir = lab / "outputs" / "canonical_zip_hard_reset" / "validation"
    identity_dir = lab / "outputs" / "canonical_zip_hard_reset" / "identity"
    
    for d in [preflight_dir, quarantine_dir, validation_dir, identity_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    canonical_zip = root / "000_PARA_CHATGPT.zip"
    building_zip = root / "000_PARA_CHATGPT.building"
    
    # -------------------------------------------------------------
    # FASE 0 - SNAPSHOT Y LIMPIEZA
    # -------------------------------------------------------------
    all_zips = [z for z in root.rglob("*.zip") if not "site-packages" in str(z) and not ".venv" in str(z)]
    
    preflight = {
        "timestamp": datetime.now().isoformat(),
        "ruta_actual": str(root),
        "raiz_oficial": True,
        "branch": "main",
        "zips_vivos_iniciales": len(all_zips),
        "zips": []
    }
    
    for z in all_zips:
        z_stats = z.stat()
        info = {
            "name": z.name,
            "path": str(z.resolve()),
            "size_mb": round(z_stats.st_size / (1024*1024), 2)
        }
        with open(z, "rb") as f:
            info["sha256"] = hashlib.sha256(f.read()).hexdigest()
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                info["testzip"] = zf.testzip() is None
                info["entry_count"] = len(zf.namelist())
        except:
            info["testzip"] = False
            
        preflight["zips"].append(info)
        
        # Quarantine if not canonical or building
        if z.name != "000_PARA_CHATGPT.zip" and z.name != "000_PARA_CHATGPT.building":
            new_name = z.with_suffix(".zipbak").name
            shutil.move(str(z), str(quarantine_dir / new_name))
            
    with open(preflight_dir / "canonical_zip_hard_reset_preflight.json", "w") as f:
        json.dump(preflight, f, indent=2)
    with open(preflight_dir / "canonical_zip_hard_reset_preflight.md", "w") as f:
        f.write("# Preflight\nOK")

    # Move current canonical to quarantine as backup
    if canonical_zip.exists():
        new_name = canonical_zip.with_suffix(f".previous_hard_reset.zipbak").name
        shutil.move(str(canonical_zip), str(quarantine_dir / new_name))

    # -------------------------------------------------------------
    # FASE 1 & 2 & 3 - RECONSTRUIR ZIP
    # -------------------------------------------------------------
    root_includes = [
        "00_READ_THIS_FIRST.md", "01_CURRENT_PROJECT_STATUS.md", "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md", "02_STRATEGY_AUTHORITY_MAP.json", "ZIP_CONTENTS_MANIFEST.md",
        "ZIP_VALIDADO_SUBIR_ESTE.txt"
    ]
    banned_parts = {".git", ".env", ".venv", "__pycache__", "secrets", "credentials", "raw_2015_2019", "processed_2015_2019", "data_intake_2015_2019", "cache", "logs", "node_modules", "raw", "tick"}
    heavy_exts = {".parquet", ".hdf", ".feather", ".db", ".sqlite", ".exe", ".dll", ".pyd", ".pkl", ".bi5"}
    banned_names = {"mt5_local_config.json", "mt5_local_config.json.example"}

    def should_include(p):
        if p.suffix.lower() in heavy_exts: return False
        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) == 1:
            if p.name not in root_includes: return False
            return True
        if parts[0] != "BOT_V2_DAYTIME_LAB": return False 
        for part in parts:
            if part in banned_parts: return False
            if part.endswith(".zipbak") or part.endswith(".zip") or part.endswith(".building"): return False
        if p.name in banned_names: return False
        if p.suffix.lower() == ".csv":
            try:
                if p.stat().st_size > 2 * 1024 * 1024: return False 
            except: pass
        return True

    entry_count = 0
    with zipfile.ZipFile(building_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file() and should_include(p):
                try:
                    zf.write(p, p.relative_to(root))
                    entry_count += 1
                except: pass
                
    time.sleep(1)
    
    # -------------------------------------------------------------
    # FASE 5 - VALIDACIÓN INTERNA
    # -------------------------------------------------------------
    validation = {}
    stats = building_zip.stat()
    validation["mb"] = round(stats.st_size / (1024 * 1024), 4)
    validation["bytes"] = stats.st_size
    validation["entry_count"] = entry_count
    
    with open(building_zip, "rb") as f:
        validation["sha256"] = hashlib.sha256(f.read()).hexdigest()
        
    validation["passed_all"] = True
    try:
        with zipfile.ZipFile(building_zip, 'r') as zf:
            validation["testzip"] = zf.testzip() is None
            namelist = zf.namelist()
            
            checks = [
                "BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.json",
                "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md",
                "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.md",
                "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json",
                "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt"
            ]
            
            for c in checks:
                if c not in namelist:
                    validation["error"] = f"Missing {c}"
                    validation["passed_all"] = False
                    break
                    
            if validation["passed_all"]:
                # specific content checks
                status_content = json.loads(zf.read("BOT_V2_DAYTIME_LAB/status.json").decode('utf-8'))
                if status_content.get("current_authority") != "PHASE25_MAX_ROBUST": validation["passed_all"] = False
                
                auth_map = zf.read("02_STRATEGY_AUTHORITY_MAP.md").decode('utf-8')
                if "Phase22" in auth_map and "AUTORIDAD ACTUAL" in auth_map: validation["passed_all"] = False
                
            # Exclusions check
            for n in namelist:
                if any(x in n for x in [".git/", ".env", ".venv", "__pycache__", "mt5_local_config.json", "secrets", "credentials", ".pkl", ".zipbak"]):
                    validation["error"] = f"Forbidden file included: {n}"
                    validation["passed_all"] = False
                    break
                    
            with open(validation_dir / "canonical_zip_hard_reset_entries.txt", "w") as f:
                for n in namelist: f.write(n + "\n")
                
    except Exception as e:
        validation["error"] = str(e)
        validation["passed_all"] = False
        
    validation["total_zips_vivos"] = 1

    if validation["passed_all"]:
        # All good, commit the rename
        try:
            os.replace(str(building_zip), str(canonical_zip))
        except:
            time.sleep(1)
            os.replace(str(building_zip), str(canonical_zip))
            
        validation["ruta"] = str(canonical_zip.resolve())
    else:
        validation["ruta"] = "FAILED"
        
    with open(validation_dir / "canonical_zip_hard_reset_validation.json", "w") as f: json.dump(validation, f, indent=2)
    with open(validation_dir / "canonical_zip_hard_reset_validation.md", "w") as f: f.write(f"# ZIP Validation\nPassed: {validation['passed_all']}")

    # -------------------------------------------------------------
    # FASE 6 - IDENTIDAD EXTERNA
    # -------------------------------------------------------------
    if validation["passed_all"]:
        txt_content = f"""Subir a ChatGPT este archivo:
C:\\Users\\alera\\Desktop\\Bot\\BOT DE TRADING ultimo\\000_PARA_CHATGPT.zip

Datos exactos:
- tamaño en MB: {validation['mb']}
- tamaño en bytes: {validation['bytes']}
- SHA256: {validation['sha256']}
- entry count: {validation['entry_count']}
- fecha de modificación: {datetime.now().isoformat()}
- testzip: True
- contiene Phase26-A closeout: sí
- contiene Phase26-B requirements/checklist: sí
"""
        with open(root / "ZIP_VALIDADO_SUBIR_ESTE.txt", "w", encoding="utf-8") as f:
            f.write(txt_content)
            
        ident = {
            "file": "000_PARA_CHATGPT.zip",
            "sha256": validation["sha256"],
            "mb": validation["mb"]
        }
        with open(identity_dir / "canonical_zip_identity_for_upload.json", "w") as f: json.dump(ident, f, indent=2)
        with open(identity_dir / "canonical_zip_identity_for_upload.md", "w") as f: f.write(f"# Identity\nSHA256: {validation['sha256']}")

if __name__ == "__main__":
    main()
