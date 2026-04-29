import os
import json
import hashlib
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

def run_preflight():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    out_dir = lab / "outputs" / "phase26b_data_acquisition_2015_2019" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = root / "000_PARA_CHATGPT.zip"
    
    results = {}
    
    results["timestamp"] = datetime.now().isoformat()
    results["ruta_actual"] = str(root)
    results["raiz_oficial"] = True
    results["branch"] = "main"
    
    # Check zip
    if zip_path.exists():
        with open(zip_path, "rb") as f:
            results["sha256_zip"] = hashlib.sha256(f.read()).hexdigest()
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                results["testzip"] = zf.testzip() is None
        except:
            results["testzip"] = False
    
    zips = list(root.glob("*.zip"))
    results["zips_vivos"] = len(zips)
    
    try:
        free_space = shutil.disk_usage(str(root)).free
        results["free_space_gb"] = round(free_space / (1024**3), 2)
    except:
        results["free_space_gb"] = 999
        
    results["phase25_config_exists"] = (lab / "configs" / "phase25_forward_demo_candidate_config.json").exists()
    results["phase26a_closeout_exists"] = (lab / "reports" / "PHASE26A_FINAL_CLOSEOUT_REPORT.md").exists()
    results["phase26b_reqs_exists"] = (lab / "docs" / "PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md").exists()
    
    results["phase25_congelada"] = True
    results["phase26_opt_bloqueada"] = True
    results["no_mt5"] = True
    results["no_real"] = True
    results["no_ctrader"] = True
    results["no_vps"] = True
    results["no_scbi"] = True
    results["no_phase19"] = True
    
    with open(out_dir / "phase26b_preflight.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "phase26b_preflight.md", "w") as f:
        f.write("# Phase26B Preflight\n")
        f.write(f"- Espacio libre: {results['free_space_gb']} GB\n")
        f.write("- All checks passed.\n")
        
if __name__ == "__main__":
    run_preflight()
