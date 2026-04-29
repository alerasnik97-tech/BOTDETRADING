import os
import shutil
import zipfile
import hashlib
import json
from pathlib import Path

def copy_and_validate():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    desktop = Path(r"C:\Users\alera\Desktop")
    canonical_zip = root / "000_PARA_CHATGPT.zip"
    upload_zip = desktop / "SUBIR_A_CHATGPT_E0C1_PHASE26_FINAL.zip"
    
    results = {}
    
    if not canonical_zip.exists():
        results["error"] = "Canonical ZIP no encontrado"
        print(json.dumps(results, indent=2))
        return
        
    results["zip_fuente_existe"] = True
    
    # Verify original
    with open(canonical_zip, "rb") as f:
        canonical_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    results["zip_fuente_sha256"] = canonical_sha256
    
    if not canonical_sha256.startswith("e0c1b55c"):
        results["error"] = f"Canonical SHA256 no coincide: {canonical_sha256}"
        print(json.dumps(results, indent=2))
        return
        
    try:
        with zipfile.ZipFile(canonical_zip, 'r') as zf:
            results["entry_count_fuente"] = len(zf.namelist())
            if "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md" not in zf.namelist():
                results["error"] = "Canonical ZIP no contiene Phase26-A closeout"
                print(json.dumps(results, indent=2))
                return
    except Exception as e:
        results["error"] = f"Error leyendo canonical ZIP: {e}"
        print(json.dumps(results, indent=2))
        return
        
    # Copy
    shutil.copy2(str(canonical_zip), str(upload_zip))
    results["copia_creada"] = upload_zip.exists()
    
    # Validate copy
    stats = upload_zip.stat()
    results["ruta_exacta_copia"] = str(upload_zip.resolve())
    results["tamaño_copia_bytes"] = stats.st_size
    results["tamaño_copia_mb"] = round(stats.st_size / (1024 * 1024), 4)
    
    with open(upload_zip, "rb") as f:
        results["sha256_copia"] = hashlib.sha256(f.read()).hexdigest()
        
    try:
        with zipfile.ZipFile(upload_zip, 'r') as zf:
            results["testzip"] = zf.testzip() is None
            namelist = zf.namelist()
            results["entry_count_copia"] = len(namelist)
            
            results["has_phase26a_closeout"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md" in namelist
            results["has_phase26b_reqs"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md" in namelist
            results["has_phase26b_chk"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md" in namelist

    except Exception as e:
        results["error"] = str(e)
        
    # Validation of 1 zip in root
    zips_in_root = [z for z in root.glob("*.zip")]
    results["zips_in_root_count"] = len(zips_in_root)
    if results["zips_in_root_count"] == 1:
        results["unique_zip_in_root"] = zips_in_root[0].name
        
    with open(root / "desktop_bypass_output.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    copy_and_validate()
