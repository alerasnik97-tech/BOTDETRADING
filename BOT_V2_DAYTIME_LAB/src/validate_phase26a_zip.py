import os
import zipfile
import json
import hashlib
from pathlib import Path

def validate_phase26a_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    zip_path = root / "000_PARA_CHATGPT.zip"
    out_dir = root / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase26_shadow_data_gap_audit" / "zip_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            results['testzip'] = zf.testzip() is None
            results['entries'] = len(zf.namelist())
            
            # Check required files
            required_files = [
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.json"
            ]
            
            missing = [f for f in required_files if f not in zf.namelist()]
            results['required_files_missing'] = missing
                
            # Check Hash of Phase 25 (must remain unchanged)
            if "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json" in zf.namelist():
                config_json = json.loads(zf.read("BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json").decode('utf-8'))
                canonical_string = json.dumps(config_json, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                calculated_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
                
                if "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in zf.namelist():
                    stored_hash = zf.read("BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt").decode('utf-8').strip()
                    results['hash_match'] = calculated_hash == stored_hash
                else:
                    results['hash_match'] = False
            else:
                results['hash_match'] = False
                
            # Check exclusions
            bad_files = [f for f in zf.namelist() if '.git/' in f or '.env' in f or 'mt5_local_config.json' in f or '.zipbak' in f or '.zip' in f or f.endswith('.parquet') or f.endswith('.hdf')]
            results['clean_exclusions'] = len(bad_files) == 0
            results['bad_files_found'] = bad_files

    except Exception as e:
        results['error'] = str(e)

    # Check zip count in root
    zip_count = len(list(root.glob("*.zip")))
    results['single_zip_in_root'] = zip_count == 1

    with open(out_dir / "phase26a_zip_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "phase26a_zip_validation.md", "w") as f:
        f.write("# PHASE 26-A ZIP INTERNAL VALIDATION\n\n")
        for k, v in results.items():
            f.write(f"- **{k}**: {v}\n")
            
    print("Validation finished.")

if __name__ == "__main__":
    validate_phase26a_zip()
