import os
import zipfile
import json
import hashlib
from pathlib import Path

def validate_zip_internals():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    zip_path = root / "000_PARA_CHATGPT.zip"
    output_dir = root / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase25_zip_content_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            results['testzip'] = zf.testzip() is None
            results['entries'] = len(zf.namelist())
            
            # Check required files
            required_files = [
                "BOT_V2_DAYTIME_LAB/docs/PHASE25_DAILY_RUNBOOK.md",
                "BOT_V2_DAYTIME_LAB/docs/PHASE25_KILL_SWITCH_POLICY.md",
                "BOT_V2_DAYTIME_LAB/docs/PHASE25_FORWARD_REVIEW_CRITERIA.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE25_FINAL_CLOSEOUT_REPORT.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE25_FINAL_CLOSEOUT_REPORT.json"
            ]
            
            missing = [f for f in required_files if f not in zf.namelist()]
            results['required_files_missing'] = missing
            
            # Check Authority Map contents
            if "02_STRATEGY_AUTHORITY_MAP.md" in zf.namelist():
                content = zf.read("02_STRATEGY_AUTHORITY_MAP.md").decode('utf-8')
                has_old_22 = "Phase 22 Logic" in content
                has_old_23 = "Phase 23 Repaired" in content
                results['authority_map_clean'] = not (has_old_22 or has_old_23)
            else:
                results['authority_map_clean'] = False
                
            # Check status.json contents
            if "BOT_V2_DAYTIME_LAB/status.json" in zf.namelist():
                try:
                    status_json = json.loads(zf.read("BOT_V2_DAYTIME_LAB/status.json").decode('utf-8'))
                    results['status_json_correct'] = (
                        status_json.get("execution_status") == "PAPER_DEMO_ONLY" and
                        status_json.get("status") == "PHASE25_FINAL_CLOSEOUT_COMPLETE_READY_FOR_PAPER_DEMO_WITH_WARNINGS"
                    )
                except:
                    results['status_json_correct'] = False
            else:
                results['status_json_correct'] = False
                
            # Check Hash
            if "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json" in zf.namelist():
                config_json = json.loads(zf.read("BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json").decode('utf-8'))
                canonical_string = json.dumps(config_json, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                calculated_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
                
                if "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in zf.namelist():
                    stored_hash = zf.read("BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt").decode('utf-8').strip()
                    results['hash_match'] = calculated_hash == stored_hash
                    results['calculated_hash'] = calculated_hash
                else:
                    results['hash_match'] = False
            else:
                results['hash_match'] = False
                
            # Check exclusions
            bad_files = [f for f in zf.namelist() if '.git/' in f or '.env' in f or 'mt5_local_config.json' in f or '.zipbak' in f or '.zip' in f]
            results['clean_exclusions'] = len(bad_files) == 0
            results['bad_files_found'] = bad_files

    except Exception as e:
        results['error'] = str(e)

    # Check zip count in root
    zip_count = len(list(root.glob("*.zip")))
    results['single_zip_in_root'] = zip_count == 1

    with open(output_dir / "final_zip_internal_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "final_zip_internal_validation.md", "w") as f:
        f.write("# FINAL ZIP INTERNAL VALIDATION\n\n")
        for k, v in results.items():
            f.write(f"- **{k}**: {v}\n")
            
    print("Validation finished.")

if __name__ == "__main__":
    validate_zip_internals()
