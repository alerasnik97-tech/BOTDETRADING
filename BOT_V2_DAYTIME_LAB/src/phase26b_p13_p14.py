import os
import json
import zipfile
import hashlib
import shutil
from pathlib import Path

def generate_f13_f14():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    
    # Phase 13: Update Masters
    proj_status = {
      "project_status": {
        "date": "2026-04-28",
        "root_status": "PHASE25_FINAL_CLOSEOUT_COMPLETE_READY_FOR_PAPER_DEMO_WITH_WARNINGS",
        "lab": "BOT_V2_DAYTIME_LAB",
        "strategies": {
          "SCBI_M5_GLOBAL": "protected_unchanged",
          "Phase18_Baseline": "daytime_baseline_protected",
          "Phase19": "INVALIDATED_AND_ARCHIVED",
          "Phase20": "benchmark_backup",
          "Phase22": "SUPERSEDED",
          "Phase24": "daytime_strong_backup",
          "Phase25": "daytime_authority_paper_demo",
          "Phase26": "READY_FOR_VALIDATION_2015_2026"
        },
        "critical_note": "Phase26B completada: Data 2015-2019 certificada con Mask. Phase26 desbloqueada para validación.",
        "mt5_touched": False,
        "real_trading_enabled": False
      }
    }
    with open(root / "01_CURRENT_PROJECT_STATUS.json", "w") as f: json.dump(proj_status, f, indent=2)
    
    with open(root / "01_CURRENT_PROJECT_STATUS.md", "w") as f: f.write("# PROJECT STATUS\nPhase25 Authority.\nPhase26B: DATA_CERTIFIED_WITH_MASK")
    
    auth_map = {
      "authority_hierarchy": {
        "daytime_primary": {
          "id": "Phase25_Max_Robust",
          "role": "daytime_authority_paper_demo",
          "status": "PHASE25_FINAL_CLOSEOUT_COMPLETE_READY_FOR_PAPER_DEMO_WITH_WARNINGS"
        },
        "research": {
            "id": "Phase26_Shadow",
            "status": "DATA_READY"
        }
      }
    }
    with open(root / "02_STRATEGY_AUTHORITY_MAP.json", "w") as f: json.dump(auth_map, f, indent=2)
    with open(root / "02_STRATEGY_AUTHORITY_MAP.md", "w") as f: f.write("# AUTHORITY MAP\nPhase 25 is authority.")
    
    lab_status = {
      "current_authority": "PHASE25_MAX_ROBUST",
      "phase26b_status": "PHASE26B_2015_2019_DATA_CERTIFIED_WITH_MASK",
      "phase26_optimization_status": "UNBLOCKED_FOR_VALIDATION"
    }
    with open(lab / "status.json", "w") as f: json.dump(lab_status, f, indent=2)
    
    # Phase 14: Canonical ZIP
    out_dir = lab / "outputs" / "phase26b_data_acquisition_2015_2019" / "zip_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    zip_building = root / "000_PARA_CHATGPT.building"
    zip_final = root / "000_PARA_CHATGPT.zip"
    
    banned_parts = {".git", ".env", ".venv", "__pycache__", "secrets", "credentials", "raw_2015_2019", "processed_2015_2019", "data_intake_2015_2019"}
    heavy_exts = {".parquet", ".hdf", ".feather", ".db", ".sqlite", ".exe", ".dll", ".pyd", ".pkl", ".bi5"}
    
    def should_include(p):
        if p.suffix.lower() in heavy_exts: return False
        parts = p.relative_to(root).parts
        for part in parts:
            if part in banned_parts: return False
            if part.endswith(".zipbak") or part.endswith(".zip"): return False
        if p.name in {"mt5_local_config.json"}: return False
        if p.suffix.lower() == ".csv":
            try:
                if p.stat().st_size > 2 * 1024 * 1024: return False 
            except: pass
        return True

    if zip_final.exists(): shutil.move(str(zip_final), str(zip_final.with_suffix(".zipbak")))

    entry_count = 0
    with zipfile.ZipFile(zip_building, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file() and should_include(p):
                try:
                    zf.write(p, p.relative_to(root))
                    entry_count += 1
                except: pass
                    
    shutil.move(str(zip_building), str(zip_final))
    
    results = {}
    stats = zip_final.stat()
    results["ruta"] = str(zip_final.resolve())
    results["mb"] = round(stats.st_size / (1024 * 1024), 4)
    results["entradas"] = entry_count
    
    with open(zip_final, "rb") as f:
        results["sha256"] = hashlib.sha256(f.read()).hexdigest()
        
    try:
        with zipfile.ZipFile(zip_final, 'r') as zf:
            results["testzip"] = zf.testzip() is None
            results["has_report"] = "BOT_V2_DAYTIME_LAB/reports/PHASE26B_EURUSD_2015_2019_DATA_CERTIFICATION_REPORT.md" in zf.namelist()
            with open(out_dir / "phase26b_zip_entries.txt", "w") as f:
                for n in zf.namelist(): f.write(n + "\n")
    except: pass
    
    results["single_zip"] = len([z for z in root.glob("*.zip") if not z.name.endswith(".zipbak")]) == 1
    
    with open(out_dir / "phase26b_zip_validation.json", "w") as f: json.dump(results, f, indent=2)
    with open(out_dir / "phase26b_zip_validation.md", "w") as f: f.write(f"# ZIP Validation\nSHA256: {results['sha256']}")

if __name__ == "__main__":
    generate_f13_f14()
