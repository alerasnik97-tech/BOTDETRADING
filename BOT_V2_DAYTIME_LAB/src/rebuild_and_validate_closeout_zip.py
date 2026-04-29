import os
import zipfile
import shutil
import hashlib
import json
from pathlib import Path

def rebuild_and_validate_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    zip_building = root / "000_PARA_CHATGPT.building"
    zip_final = root / "000_PARA_CHATGPT.zip"
    out_dir = lab / "outputs" / "phase26a_final_closeout" / "zip_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Explicit includes (at root)
    root_includes = [
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "ZIP_CONTENTS_MANIFEST.md"
    ]
    
    banned_parts = {
        ".git", ".env", ".venv", "__pycache__", "secrets", "credentials", "tokens", "keys", "pem", "node_modules", "raw", "tick", "data_intake", "data", "logs"
    }
    banned_names = {
        "mt5_local_config.json", "mt5_local_config.json.example"
    }
    heavy_exts = {".parquet", ".hdf", ".feather", ".db", ".sqlite", ".exe", ".dll", ".pyd", ".pkl"}

    def should_include(p):
        if p.suffix.lower() in heavy_exts: return False
        
        rel = p.relative_to(root)
        parts = rel.parts
        
        if len(parts) == 1:
            if p.name not in root_includes: return False
            return True
            
        if parts[0] != "BOT_V2_DAYTIME_LAB":
            return False 
            
        for part in parts:
            if part in banned_parts: return False
            if part.endswith(".zipbak") or part.endswith(".zip"): return False
            
        if p.name in banned_names: return False
        
        if p.suffix.lower() == ".csv":
            try:
                if p.stat().st_size > 2 * 1024 * 1024: return False 
            except:
                pass
                
        return True

    # Backup existing
    if zip_final.exists():
        backup_name = zip_final.with_suffix(f".previous_closeout.zipbak")
        shutil.move(str(zip_final), str(backup_name))

    entry_count = 0
    with zipfile.ZipFile(zip_building, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file() and should_include(p):
                try:
                    rel_path = p.relative_to(root)
                    zf.write(p, rel_path)
                    entry_count += 1
                except Exception as e:
                    print(f"Skipping {p.name}: {e}")
                    
    # Neutralize other zips
    for p in root.rglob("*.zip"):
        if p.name != "000_PARA_CHATGPT.building":
            try:
                p.rename(p.with_suffix(".zipbak"))
            except:
                pass

    shutil.move(str(zip_building), str(zip_final))
    print(f"Build complete. Entries: {entry_count}")

    # Validation
    results = {}
    stats = zip_final.stat()
    results["ruta"] = str(zip_final.resolve())
    results["mb"] = round(stats.st_size / (1024 * 1024), 4)
    results["entradas"] = entry_count
    
    with open(zip_final, "rb") as f:
        results["sha256"] = hashlib.sha256(f.read()).hexdigest()
        
    try:
        with zipfile.ZipFile(zip_final, 'r') as zf:
            results["testzip"] = zf.testzip() is None
            namelist = zf.namelist()
            with open(out_dir / "phase26a_final_zip_entries.txt", "w") as f:
                for n in namelist: f.write(n + "\n")
            
            # checks
            results["has_closeout"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26A_FINAL_CLOSEOUT_REPORT.md" in namelist
            results["has_reqs"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md" in namelist
            results["has_chk"] = "BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md" in namelist
            results["has_audit"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.md" in namelist
            results["has_outputs"] = any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase26_shadow_data_gap_audit/") for n in namelist)
            results["has_phase25_config"] = "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json" in namelist
            results["has_phase25_closeout"] = "BOT_V2_DAYTIME_LAB/reports/PHASE25_FINAL_CLOSEOUT_REPORT.md" in namelist
            results["heavy_files"] = any(".pkl" in n or ".parquet" in n for n in namelist)
            results["secrets_found"] = any(".env" in n or "mt5_local_config.json" in n for n in namelist)
            
    except Exception as e:
        results["error"] = str(e)
        
    results["single_zip"] = len(list(root.glob("*.zip"))) == 1
    
    with open(out_dir / "phase26a_final_zip_validation.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "phase26a_final_zip_validation.md", "w", encoding="utf-8") as f:
        f.write("# PHASE26-A FINAL ZIP VALIDATION\n")
        for k, v in results.items():
            f.write(f"- {k}: {v}\n")
            
if __name__ == "__main__":
    rebuild_and_validate_zip()
