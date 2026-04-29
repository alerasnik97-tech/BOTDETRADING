
import os
import zipfile
import time
from pathlib import Path

def create_canonical_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    new_zip_path = root / "000_PARA_CHATGPT.new.zip"
    final_zip_path = root / "000_PARA_CHATGPT.zip"
    
    include_dirs = ['BOT_V2_DAYTIME_LAB', 'SCBI_M5_GLOBAL', 'docs', 'configs']
    include_subdirs = {
        'phase23_consistency_repair', 
        'phase24_controlled_optimization_2015_2026', 
        'phase25_max_robust_plateau',
        'canonical_zip_authority_repair'
    }
    
    exclude_parts = {'.git', '.venv', '__pycache__', 'raw_data', 'data', 'results', 'research_lab', 'ARCHIVE_SUPERSEDED', '.ipynb_checkpoints'}
    exclude_exts = {'.zip', '.csv', '.parquet', '.log', '.png', '.jpg', '.pdf', '.exe', '.dll', '.bin'}
    
    print(f"Creating NEW CANONICAL ZIP at {new_zip_path}...")
    
    try:
        with zipfile.ZipFile(new_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in root.iterdir():
                if f.is_file() and (f.suffix == '.md' or f.suffix == '.json'):
                    zf.write(f, f.name)
            
            for inc_dir in include_dirs:
                dir_path = root / inc_dir
                if not dir_path.exists(): continue
                for file_path in dir_path.rglob('*'):
                    if any(part in exclude_parts for part in file_path.parts): continue
                    if file_path.suffix.lower() in exclude_exts:
                        is_audit_csv = any(sub in str(file_path) for sub in include_subdirs) and file_path.suffix.lower() == '.csv'
                        if is_audit_csv and file_path.stat().st_size < 200000: pass
                        else: continue
                    if file_path.is_file():
                        if file_path.name.lower() in {'.env', 'secrets', 'credentials', 'mt5_local_config.json', 'tokens', 'key', 'pem'}: continue
                        zf.write(file_path, file_path.relative_to(root))
                        
        print("NEW ZIP Created. Validating...")
        time.sleep(2) # Wait for handles to clear
        
        # Validation
        zf_check = zipfile.ZipFile(new_zip_path, 'r')
        test_res = zf_check.testzip()
        namelist = zf_check.namelist()
        zf_check.close() # Explicit close
        
        if test_res is not None:
            raise Exception("ZIP self-test failed.")
        
        print(f"Validation successful. Entries: {len(namelist)}")
        time.sleep(2) # Final wait
        
        # Final Swap
        if final_zip_path.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = root / f"000_PARA_CHATGPT.backup_{ts}.zip"
            os.rename(final_zip_path, backup_path)
            print(f"Backup created: {backup_path.name}")
        
        os.rename(new_zip_path, final_zip_path)
        print(f"CANONICAL AUTHORITY REPAIRED: {final_zip_path.name}")
            
    except Exception as e:
        print(f"PROCESS ERROR: {e}")
        # Only delete if it's NOT the final ZIP (to avoid losing work if rename partially worked)
        if new_zip_path.exists() and not final_zip_path.exists():
            # os.remove(new_zip_path)
            pass

if __name__ == "__main__":
    create_canonical_zip()
