import os
import zipfile
import hashlib
import json
from pathlib import Path

def get_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def verify_local_vs_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    zip_path = root / "000_PARA_CHATGPT.zip"
    output_dir = root / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase25_zip_content_verification"
    
    required_files = [
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "00_READ_THIS_FIRST.md",
        "ZIP_CONTENTS_MANIFEST.md",
        "BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md",
        "BOT_V2_DAYTIME_LAB/status.json",
        "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json",
        "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt",
        "BOT_V2_DAYTIME_LAB/reports/PHASE25_FINAL_CLOSEOUT_REPORT.md",
        "BOT_V2_DAYTIME_LAB/reports/PHASE25_FINAL_CLOSEOUT_REPORT.json",
        "BOT_V2_DAYTIME_LAB/docs/PHASE25_DAILY_RUNBOOK.md",
        "BOT_V2_DAYTIME_LAB/docs/PHASE25_KILL_SWITCH_POLICY.md",
        "BOT_V2_DAYTIME_LAB/docs/PHASE25_FORWARD_REVIEW_CRITERIA.md"
    ]
    
    results = []
    
    zip_exists = zip_path.exists()
    if zip_exists:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zip_namelist = zf.namelist()
                for rel_path in required_files:
                    local_path = root / rel_path
                    local_exists = local_path.exists()
                    local_sha = ""
                    if local_exists:
                        local_content = local_path.read_bytes()
                        # Normalize line endings for hash comparison if needed, but let's stick to raw bytes first
                        local_sha = get_sha256(local_content)
                    
                    z_exists = rel_path in zip_namelist
                    zip_sha = ""
                    if z_exists:
                        zip_content = zf.read(rel_path)
                        zip_sha = get_sha256(zip_content)
                        
                    results.append({
                        "file": rel_path,
                        "local_exists": local_exists,
                        "zip_exists": z_exists,
                        "local_sha256": local_sha,
                        "zip_sha256": zip_sha,
                        "match": local_sha == zip_sha and local_exists and z_exists
                    })
        except Exception as e:
            print(f"Error reading zip: {e}")
    else:
        print("ZIP does not exist.")
        
    with open(output_dir / "phase25_local_vs_zip_verification.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(output_dir / "phase25_local_vs_zip_verification.md", "w") as f:
        f.write("# Local vs ZIP Verification\n\n")
        f.write("| File | Local Exists | ZIP Exists | Match |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            match_str = "YES" if r["match"] else "NO"
            f.write(f"| {r['file']} | {r['local_exists']} | {r['zip_exists']} | {match_str} |\n")
            
if __name__ == "__main__":
    verify_local_vs_zip()
