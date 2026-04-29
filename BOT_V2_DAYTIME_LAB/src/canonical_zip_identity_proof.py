import os
import zipfile
import hashlib
import json
from datetime import datetime
from pathlib import Path

def prove_canonical_zip_identity():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    zip_path = root / "000_PARA_CHATGPT.zip"
    out_dir = root / "BOT_V2_DAYTIME_LAB" / "outputs" / "canonical_zip_identity_proof"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if not zip_path.exists():
        results["error"] = "ZIP no encontrado"
        with open(out_dir / "canonical_zip_identity_proof.json", "w") as f:
            json.dump(results, f, indent=2)
        return
        
    stats = zip_path.stat()
    results["ruta_absoluta"] = str(zip_path.resolve())
    results["tamaño_bytes"] = stats.st_size
    results["tamaño_mb"] = round(stats.st_size / (1024 * 1024), 4)
    results["modificado"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
    
    with open(zip_path, "rb") as f:
        bytes = f.read()
        results["sha256"] = hashlib.sha256(bytes).hexdigest()
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            results["testzip"] = zf.testzip() is None
            namelist = zf.namelist()
            results["entradas_totales"] = len(namelist)
            
            results["primeras_30"] = namelist[:30]
            results["ultimas_30"] = namelist[-30:]
            
            with open(out_dir / "canonical_zip_entries.txt", "w") as f:
                for n in namelist: f.write(n + "\n")
                
            results["count_phase26"] = sum(1 for n in namelist if "phase26" in n)
            results["count_PHASE26"] = sum(1 for n in namelist if "PHASE26" in n)
            results["count_shadow"] = sum(1 for n in namelist if "phase26_shadow_data_gap_audit" in n)
            results["count_PHASE26A_DATA_GAP"] = sum(1 for n in namelist if "PHASE26A_DATA_GAP" in n)
            
            # Tarea 2: validate files exactly
            exact_files = [
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.md",
                "BOT_V2_DAYTIME_LAB/reports/PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.json"
            ]
            exact_dirs = [
                "BOT_V2_DAYTIME_LAB/outputs/phase26_shadow_data_gap_audit/data_inventory/",
                "BOT_V2_DAYTIME_LAB/outputs/phase26_shadow_data_gap_audit/data_coverage/",
                "BOT_V2_DAYTIME_LAB/outputs/phase26_shadow_data_gap_audit/certification_plan/",
                "BOT_V2_DAYTIME_LAB/outputs/phase26_shadow_data_gap_audit/zip_validation/"
            ]
            
            results["exact_files_found"] = {f: f in namelist for f in exact_files}
            
            # check dirs by seeing if any file starts with that path
            dir_found = {}
            for d in exact_dirs:
                dir_found[d] = any(n.startswith(d) for n in namelist)
            results["exact_dirs_found"] = dir_found
            
            all_phase26a_present = all(results["exact_files_found"].values()) and all(dir_found.values())
            results["phase26a_fully_included"] = all_phase26a_present
            
            results["phase25_config"] = "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json" in namelist
            results["phase25_hash"] = "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in namelist
            
            results["heavy_pkl_included"] = any(".pkl" in n for n in namelist)
            
    except Exception as e:
        results["error"] = str(e)
        
    with open(out_dir / "canonical_zip_identity_proof.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "canonical_zip_identity_proof.md", "w", encoding='utf-8') as f:
        f.write("# CANONICAL ZIP IDENTITY PROOF\n")
        f.write(f"- Ruta: {results['ruta_absoluta']}\n")
        f.write(f"- Tamaño: {results['tamaño_mb']} MB\n")
        f.write(f"- Modificado: {results['modificado']}\n")
        f.write(f"- Entradas: {results.get('entradas_totales', 'ERROR')}\n")
        f.write(f"- SHA256: {results.get('sha256', 'ERROR')}\n")
        f.write(f"- Phase26A Completo: {results.get('phase26a_fully_included', False)}\n")

if __name__ == "__main__":
    prove_canonical_zip_identity()
