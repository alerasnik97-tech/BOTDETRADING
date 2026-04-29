import os
import shutil
import zipfile
import hashlib
import json
from pathlib import Path

def force_create_visible_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    canonical_zip = root / "000_PARA_CHATGPT.zip"
    upload_zip = root / "SUBIR_A_CHATGPT_PHASE26A_771efe.zip"
    
    results = {}
    
    if not canonical_zip.exists():
        results["error"] = "Canonical ZIP no encontrado"
        print(json.dumps(results, indent=2))
        return
        
    results["zip_fuente_existe"] = True
    
    # Verify original
    with open(canonical_zip, "rb") as f:
        bytes = f.read()
        canonical_sha256 = hashlib.sha256(bytes).hexdigest()
        
    results["zip_fuente_sha256"] = canonical_sha256
    
    if not canonical_sha256.startswith("771efe"):
        results["error"] = f"Canonical SHA256 no coincide: {canonical_sha256}"
        print(json.dumps(results, indent=2))
        return
        
    try:
        with zipfile.ZipFile(canonical_zip, 'r') as zf:
            namelist = zf.namelist()
            if not "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md" in namelist:
                results["error"] = "Canonical ZIP no contiene PHASE26A_FINAL_CLOSEOUT_REPORT.md"
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
        bytes = f.read()
        results["sha256_copia"] = hashlib.sha256(bytes).hexdigest()
        
    try:
        with zipfile.ZipFile(upload_zip, 'r') as zf:
            results["testzip"] = zf.testzip() is None
            namelist = zf.namelist()
            results["entry_count"] = len(namelist)
            
            results["has_phase26a_closeout"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md" in namelist
            results["has_phase26b_reqs"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md" in namelist
            results["has_phase26b_chk"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md" in namelist

    except Exception as e:
        results["error"] = str(e)
        
    with open(root / "force_visible_zip_output.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    force_create_visible_zip()
