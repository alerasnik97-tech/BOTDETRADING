import os
import json
import zipfile
import hashlib
import shutil
import time
from pathlib import Path

def generate_f13_f14():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    
    out_dir = lab / "outputs" / "phase26b_data_acquisition_2015_2019" / "zip_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    zip_building = root / "000_PARA_CHATGPT.building"
    zip_final = root / "000_PARA_CHATGPT.zip"
    
    banned_parts = {".git", ".env", ".venv", "__pycache__", "secrets", "credentials", "raw_2015_2019", "processed_2015_2019", "data_intake_2015_2019"}
    heavy_exts = {".parquet", ".hdf", ".feather", ".db", ".sqlite", ".exe", ".dll", ".pyd", ".pkl", ".bi5"}
    
    def should_include(p):
        if p.suffix.lower() in heavy_exts: return False
        parts = p.relative_to(root).parts
        for part in parts:
            if part in banned_parts: return False
            if part.endswith(".zipbak") or part.endswith(".zip") or part.endswith(".building"): return False
        if p.name in {"mt5_local_config.json"}: return False
        if p.suffix.lower() == ".csv":
            try:
                if p.stat().st_size > 2 * 1024 * 1024: return False 
            except: pass
        return True

    if zip_final.exists(): 
        shutil.move(str(zip_final), str(zip_final.with_suffix(".zipbak")))

    entry_count = 0
    with zipfile.ZipFile(zip_building, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file() and should_include(p):
                try:
                    zf.write(p, p.relative_to(root))
                    entry_count += 1
                except: pass
                    
    time.sleep(2) # ensure lock is released
    if zip_building.exists():
        shutil.move(str(zip_building), str(zip_final))
    
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
            results["has_report"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26B_EURUSD_2015_2019_DATA_CERTIFICATION_REPORT.md" in zf.namelist()
            with open(out_dir / "phase26b_zip_entries.txt", "w") as f:
                for n in zf.namelist(): f.write(n + "\n")
    except: pass
    
    results["single_zip"] = len([z for z in root.glob("*.zip") if not z.name.endswith(".zipbak") and not z.name.endswith(".building")]) == 1
    
    with open(out_dir / "phase26b_zip_validation.json", "w") as f: json.dump(results, f, indent=2)
    with open(out_dir / "phase26b_zip_validation.md", "w") as f: f.write(f"# ZIP Validation\nSHA256: {results['sha256']}")

if __name__ == "__main__":
    generate_f13_f14()
