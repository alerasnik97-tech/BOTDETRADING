import os
import zipfile
import shutil
from pathlib import Path

def repair_canonical_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    zip_building = root / "000_PARA_CHATGPT.building"
    zip_final = root / "000_PARA_CHATGPT.zip"
    
    print("Local Phase26-A exists. Proceeding to build lean ZIP.")

    # Explicit includes (at root)
    root_includes = [
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "ZIP_CONTENTS_MANIFEST.md"
    ]
    
    # We will include BOT_V2_DAYTIME_LAB completely, EXCEPT data and cache
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
        
        # If it's a file in root, check root_includes
        if len(parts) == 1:
            if p.name not in root_includes: return False
            return True
            
        # If it's inside BOT_V2_DAYTIME_LAB, include it minus banned
        if parts[0] != "BOT_V2_DAYTIME_LAB":
            return False # Drop anything else like legacy_archive, results, .tmp, etc.
            
        for part in parts:
            if part in banned_parts: return False
            if part.endswith(".zipbak") or part.endswith(".zip"): return False
            
        if p.name in banned_names: return False
        
        # Exclude heavy CSVs
        if p.suffix.lower() == ".csv":
            try:
                if p.stat().st_size > 2 * 1024 * 1024: return False # 2MB limit
            except:
                pass
                
        return True

    # Backup existing
    if zip_final.exists():
        backup_name = zip_final.with_suffix(f".previous_phase26a.zipbak")
        shutil.move(str(zip_final), str(backup_name))

    entry_count = 0
    with zipfile.ZipFile(zip_final, 'w', zipfile.ZIP_DEFLATED) as zf:
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
        if p.name != "000_PARA_CHATGPT.zip":
            try:
                p.rename(p.with_suffix(".zipbak"))
            except:
                pass
                
    print(f"Build complete. Entries: {entry_count}")
    print("SINGLE ZIP AUTHORITY ESTABLISHED: 000_PARA_CHATGPT.zip")

if __name__ == "__main__":
    repair_canonical_zip()
