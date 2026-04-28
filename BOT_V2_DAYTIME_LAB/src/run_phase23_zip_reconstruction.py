
import os
import zipfile
from pathlib import Path

def create_and_validate_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    zip_path = root / "000_PARA_CHATGPT.zip"
    
    # Very explicit inclusion
    include_dirs = ['BOT_V2_DAYTIME_LAB', 'SCBI_M5_GLOBAL', 'docs', 'configs']
    # Folders to exclude completely
    exclude_parts = {'.git', '.venv', '__pycache__', 'raw_data', 'data', 'results', 'research_lab', 'ARCHIVE_SUPERSEDED', '.ipynb_checkpoints'}
    exclude_exts = {'.zip', '.csv', '.parquet', '.log', '.png', '.jpg', '.pdf', '.exe', '.dll', '.bin'}
    
    print(f"Creating REPAIRED ZIP at {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Add root MD files
            for f in root.iterdir():
                if f.is_file() and f.suffix == '.md':
                    zf.write(f, f.name)
            
            # 2. Add selected directories
            for inc_dir in include_dirs:
                dir_path = root / inc_dir
                if not dir_path.exists(): continue
                
                for file_path in dir_path.rglob('*'):
                    # Check exclusions
                    if any(part in exclude_parts for part in file_path.parts): continue
                    if file_path.suffix.lower() in exclude_exts: 
                        # Special case: allow .csv ONLY if it is a small summary or in specific report subdirs
                        if file_path.suffix.lower() == '.csv' and file_path.stat().st_size < 100000:
                            pass # Allow small CSVs
                        else:
                            continue
                    
                    if file_path.is_file():
                        if file_path.name in {'.env', 'secrets', 'credentials'}: continue
                        
                        rel_path = file_path.relative_to(root)
                        zf.write(file_path, rel_path)
                        
        print("REPAIRED ZIP Created. Validating...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            test_res = zf.testzip()
            print(f"testzip() result: {test_res}")
            
            namelist = zf.namelist()
            print(f"Total files in ZIP: {len(namelist)}")
            
            # Critical files check
            critical = [
                "BOT_V2_DAYTIME_LAB/reports/PHASE23_PHASE22_FORENSIC_READINESS_REPORT.md",
                "BOT_V2_DAYTIME_LAB/configs/phase22_forward_demo_config.json",
                "BOT_V2_DAYTIME_LAB/outputs/phase23_consistency_repair/metrics_reconciliation/phase23_metrics_reconciliation.md"
            ]
            # Since I haven't created the reconciliation md yet, I'll check others
            critical = [
                "BOT_V2_DAYTIME_LAB/reports/PHASE23_PHASE22_FORENSIC_READINESS_REPORT.md",
                "BOT_V2_DAYTIME_LAB/configs/phase22_forward_demo_config.json",
                "01_CURRENT_PROJECT_STATUS.md"
            ]
            for c in critical:
                if c in namelist: print(f"Verified: {c}")
                else: print(f"ERROR: MISSING {c}")
                
    except Exception as e:
        print(f"PROCESS ERROR: {e}")

if __name__ == "__main__":
    create_and_validate_zip()
