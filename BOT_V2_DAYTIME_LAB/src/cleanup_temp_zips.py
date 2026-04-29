import os
import zipfile
import hashlib
import json
import shutil
from pathlib import Path

def cleanup_temp_zips():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    canonical_zip = root / "000_PARA_CHATGPT.zip"
    
    out_dir = lab / "outputs" / "canonical_zip_cleanup"
    quarantine_dir = out_dir / "quarantine"
    out_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "canonical_validated": False,
        "sha256": "",
        "entry_count": 0,
        "testzip": False,
        "temp_zips_found": [],
        "actions_taken": [],
        "final_zip_count": 0,
        "unique_zip": "",
        "no_permanent_delete": True,
        "no_mt5": True,
        "no_real": True,
        "no_commit": True,
        "no_push": True
    }
    
    # 1. Validar ZIP Canónico
    if not canonical_zip.exists():
        results["error"] = "Canonical ZIP no encontrado"
        with open(out_dir / "post_upload_cleanup.json", "w") as f:
            json.dump(results, f, indent=2)
        return
        
    with open(canonical_zip, "rb") as f:
        bytes = f.read()
        canonical_sha256 = hashlib.sha256(bytes).hexdigest()
        
    if canonical_sha256 != "771efe023d9888673a2bd2e4a58362de9cb5fb08b5a0b7152cc805fd8c255782":
        results["error"] = f"Canonical SHA256 incorrecto: {canonical_sha256}"
        with open(out_dir / "post_upload_cleanup.json", "w") as f:
            json.dump(results, f, indent=2)
        return
        
    results["sha256"] = canonical_sha256
    
    try:
        with zipfile.ZipFile(canonical_zip, 'r') as zf:
            results["testzip"] = zf.testzip() is None
            namelist = zf.namelist()
            results["entry_count"] = len(namelist)
            
            checks = [
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.json",
                "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md",
                "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md",
                "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json",
                "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt"
            ]
            
            for c in checks:
                if c not in namelist:
                    results["error"] = f"Falta {c} en ZIP canónico"
                    with open(out_dir / "post_upload_cleanup.json", "w") as f:
                        json.dump(results, f, indent=2)
                    return
                    
            if results["entry_count"] != 784:
                results["error"] = f"Entry count no es 784: {results['entry_count']}"
                with open(out_dir / "post_upload_cleanup.json", "w") as f:
                    json.dump(results, f, indent=2)
                return
                
    except Exception as e:
        results["error"] = f"Error validando ZIP: {e}"
        with open(out_dir / "post_upload_cleanup.json", "w") as f:
            json.dump(results, f, indent=2)
        return
        
    results["canonical_validated"] = True
    
    # 2. Cleanup Temp Zips
    all_zips = list(root.rglob("*.zip"))
    temp_zips = [z for z in all_zips if z.resolve() != canonical_zip.resolve() and not 'site-packages' in str(z) and not '.venv' in str(z)]
    
    for z in temp_zips:
        results["temp_zips_found"].append(str(z.name))
        
        # rename to .zipbak and move to quarantine
        new_name = z.with_suffix(".zipbak").name
        dest = quarantine_dir / new_name
        
        try:
            shutil.move(str(z), str(dest))
            results["actions_taken"].append(f"Moved {z.name} to quarantine as {new_name}")
        except Exception as e:
            results["actions_taken"].append(f"Error moving {z.name}: {e}")
            
    # 3. Final Validation
    final_zips = [z for z in root.rglob("*.zip") if not 'site-packages' in str(z) and not '.venv' in str(z)]
    results["final_zip_count"] = len(final_zips)
    
    if len(final_zips) == 1:
        results["unique_zip"] = str(final_zips[0].name)
        
    with open(out_dir / "post_upload_cleanup.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "post_upload_cleanup.md", "w", encoding='utf-8') as f:
        f.write("# POST UPLOAD CLEANUP\n")
        f.write(f"- Validado: {results['canonical_validated']}\n")
        f.write(f"- SHA256: {results['sha256']}\n")
        f.write(f"- Único ZIP: {results['unique_zip']}\n")
        for a in results["actions_taken"]:
            f.write(f"- Acción: {a}\n")

if __name__ == "__main__":
    cleanup_temp_zips()
