import os
import zipfile
import hashlib
import json
from datetime import datetime
from pathlib import Path

def debug_zip_identity():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    zip_path = root / "000_PARA_CHATGPT.zip"
    
    results = {}
    
    if not zip_path.exists():
        results["error"] = "ZIP no encontrado"
        print(json.dumps(results, indent=2))
        return
        
    stats = zip_path.stat()
    results["ruta_absoluta"] = str(zip_path.resolve())
    results["tamaño_bytes"] = stats.st_size
    results["tamaño_mb"] = round(stats.st_size / (1024 * 1024), 4)
    results["modificado"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
    
    with open(zip_path, "rb") as f:
        bytes = f.read()
        results["sha256"] = hashlib.sha256(bytes).hexdigest()
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            results["testzip"] = zf.testzip() is None
            namelist = zf.namelist()
            results["entry_count"] = len(namelist)
            
            # Check for Phase26A closeout docs
            results["has_phase26a_closeout"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md" in namelist
            results["has_phase26b_reqs"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md" in namelist
            results["has_phase26b_chk"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md" in namelist
            results["has_outputs_closeout"] = any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase26a_final_closeout") for n in namelist)

    except Exception as e:
        results["error"] = str(e)

    # Validate only one zip is alive
    results["total_zips_vivos"] = len(list(root.glob("*.zip")))
    
    with open(root / "debug_zip_identity_output.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    debug_zip_identity()
