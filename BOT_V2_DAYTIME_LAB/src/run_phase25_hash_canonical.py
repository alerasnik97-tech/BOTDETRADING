
import json
import hashlib
from pathlib import Path

def validate_and_recalc_hash():
    config_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\configs\phase25_forward_demo_candidate_config.json")
    hash_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\configs\phase25_forward_demo_candidate_config_hash.txt")
    evidence_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase25_final_closeout\hash_validation")
    
    if not config_path.exists():
        print("PHASE25_CLOSEOUT_REQUIRES_REPAIR: Config missing.")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    # Canonical serialization
    canonical_string = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    
    with open(evidence_dir / "phase25_config_canonical_string.txt", 'w', encoding='utf-8') as f:
        f.write(canonical_string)
        
    new_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
    
    old_hash = ""
    if hash_path.exists():
        old_hash = hash_path.read_text().strip()
        
    with open(hash_path, 'w') as f:
        f.write(new_hash)
        
    print(f"Hash calculation complete.")
    print(f"Old hash: {old_hash}")
    print(f"New hash: {new_hash}")
    print(f"Matches: {old_hash == new_hash}")

if __name__ == "__main__":
    validate_and_recalc_hash()
