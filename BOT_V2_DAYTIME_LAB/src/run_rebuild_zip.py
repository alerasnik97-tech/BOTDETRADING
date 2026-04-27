
import os
import shutil
import zipfile
from pathlib import Path

def rebuild_zip():
    base = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    staging = base / "_zip_staging_phase18"
    zip_file_path = base / "000_PARA_CHATGPT.zip"

    # Phase 2: Remove old ZIP
    if zip_file_path.exists():
        zip_file_path.unlink()
        print(f"Removed existing {zip_file_path.name}")

    # Phase 3: Create staging
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"Created staging: {staging}")

    to_copy = [
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "03_OBSOLETE_AND_SUPERSEDED_INDEX.md",
        "03_OBSOLETE_AND_SUPERSEDED_INDEX.json",
        "ZIP_CONTENTS_MANIFEST.md",
        "BOT_V2_DAYTIME_LAB/reports",
        "BOT_V2_DAYTIME_LAB/src",
        "BOT_V2_DAYTIME_LAB/outputs/phase18_h1_fractal_sweep",
        "BOT_V2_DAYTIME_LAB/outputs/manual_edge_alignment",
        "BOT_V2_DAYTIME_LAB/data/manual_edge_alignment",
        "REPORTS/infra_audits",
        "REPORTS/engine_safety",
        "REPORTS/vps_readiness",
        "STRATEGIES",
        "VPS_READINESS"
    ]

    for item in to_copy:
        src = base / item
        if src.exists():
            dest = staging / item
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)
            # print(f"Copied {item}")

    # Exclusion cleanup
    exclude_patterns = [
        ".git", ".venv", "__pycache__", ".pyc", "cache", "logs", "temp", 
        "_staging", "ARCHIVE_SUPERSEDED", "mt5_local_config.json", ".env", 
        "secrets", "credentials", ".key", ".pem"
    ]
    
    for root, dirs, files in os.walk(staging, topdown=False):
        for name in files:
            file_path = Path(root) / name
            if any(p in name for p in exclude_patterns) or name.endswith('.zip'):
                file_path.unlink()
        for name in dirs:
            dir_path = Path(root) / name
            if any(p in name for p in exclude_patterns):
                shutil.rmtree(dir_path)

    # Phase 4: Compress
    print(f"Compressing into {zip_file_path.name}...")
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(staging):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(staging)
                zipf.write(file_path, arcname)

    # Phase 5: Validation
    print("Validating ZIP...")
    if not zip_file_path.exists():
        raise Exception("ZIP file not found after compression.")
    
    size = zip_file_path.stat().st_size
    print(f"ZIP Size: {size / 1024:.2f} KB")
    if size == 0:
        raise Exception("ZIP file is empty.")

    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        bad_file = zipf.testzip()
        if bad_file:
            raise Exception(f"ZIP file is corrupted. First bad file: {bad_file}")
        
        contents = zipf.namelist()
        required = [
            "BOT_V2_DAYTIME_LAB/reports/PHASE18_H1_FRACTAL_SWEEP_ALIGNMENT_REPORT.md",
            "BOT_V2_DAYTIME_LAB/reports/PHASE18_H1_FRACTAL_SWEEP_ALIGNMENT_REPORT.json",
            "BOT_V2_DAYTIME_LAB/src/phase18_h1_fractal_sweep.py",
            "BOT_V2_DAYTIME_LAB/src/phase18_first_3m_choch.py",
            "00_READ_THIS_FIRST.md",
            "ZIP_CONTENTS_MANIFEST.md"
        ]
        # Check outputs folder existence in namelist
        outputs_found = any(c.startswith("BOT_V2_DAYTIME_LAB/outputs/phase18_h1_fractal_sweep/") for c in contents)
        
        for req in required:
            if req not in contents:
                raise Exception(f"Missing required file in ZIP: {req}")
        
        if not outputs_found:
            raise Exception("Missing outputs/phase18_h1_fractal_sweep/ in ZIP")

        forbidden = [".git", ".venv", "__pycache__", "mt5_local_config.json", ".env", "DATA/", "ARCHIVE_SUPERSEDED/"]
        for f in forbidden:
            if any(c.startswith(f) or f in c for c in contents):
                 # Data/ and ARCHIVE_SUPERSEDED/ are directories
                 print(f"WARNING: Found potentially forbidden content: {f}")

    # Phase 6: Cleanup
    shutil.rmtree(staging)
    print("Cleanup complete.")

if __name__ == "__main__":
    rebuild_zip()
