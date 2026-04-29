import os
import zipfile
import shutil
import hashlib
import json
import time
from pathlib import Path

def rebuild_and_validate_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    zip_building = root / "000_PARA_CHATGPT.building"
    zip_final = root / "000_PARA_CHATGPT.zip"
    out_dir = lab / "outputs" / "phase26b_data_acquisition_2015_2019" / "zip_validation"
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
    heavy_exts = {".parquet", ".hdf", ".feather", ".db", ".sqlite", ".exe", ".dll", ".pyd", ".pkl", ".bi5"}

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
            if part.endswith(".zipbak") or part.endswith(".zip") or part.endswith(".building"): return False
            
        if p.name in banned_names: return False
        
        if p.suffix.lower() == ".csv":
            try:
                if p.stat().st_size > 2 * 1024 * 1024: return False 
            except:
                pass
                
        return True

    # Backup existing
    if zip_final.exists():
        backup_name = zip_final.with_suffix(f".previous_phase26b.zipbak")
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
                    pass
                    
    # Force rename with retry
    for _ in range(5):
        try:
            os.replace(str(zip_building), str(zip_final))
            break
        except:
            time.sleep(1)

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
            with open(out_dir / "phase26b_zip_entries.txt", "w") as f:
                for n in namelist: f.write(n + "\n")
            
            # checks
            results["has_phase26b_report"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26B_EURUSD_2015_2019_DATA_CERTIFICATION_REPORT.md" in namelist
            
    except Exception as e:
        results["error"] = str(e)
        
    results["single_zip"] = len([z for z in root.glob("*.zip") if not z.name.endswith(".zipbak") and not z.name.endswith(".building")]) == 1
    
    with open(out_dir / "phase26b_zip_validation.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "phase26b_zip_validation.md", "w", encoding="utf-8") as f:
        f.write("# PHASE26-B FINAL ZIP VALIDATION\n")
        for k, v in results.items():
            f.write(f"- {k}: {v}\n")
            
if __name__ == "__main__":
    rebuild_and_validate_zip()
