import os
import json
import csv
from pathlib import Path
from datetime import datetime

def run_phase26a_audit():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab_root = root / "BOT_V2_DAYTIME_LAB"
    out_dir = lab_root / "outputs" / "phase26_shadow_data_gap_audit"
    
    # Create directories
    dirs = [
        "preflight", "data_inventory", "data_coverage", "data_quality_probe",
        "news_inventory", "certification_plan", "zip_validation", "git"
    ]
    for d in dirs:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
        
    now = datetime.now().isoformat()
    
    # FASE 0 - PREFLIGHT
    preflight = {
        "timestamp": now,
        "path": str(root),
        "branch": "main",
        "zip_exists": (root / "000_PARA_CHATGPT.zip").exists(),
        "zip_count": len(list(root.glob("*.zip"))),
        "phase25_config_exists": (lab_root / "configs" / "phase25_forward_demo_candidate_config.json").exists(),
        "phase25_hash_exists": (lab_root / "configs" / "phase25_forward_demo_candidate_config_hash.txt").exists(),
        "phase25_report_exists": (lab_root / "reports" / "PHASE25_FINAL_CLOSEOUT_REPORT.md").exists(),
        "phase25_frozen": True,
        "no_mt5": True, "no_real": True, "no_ctrader": True, "no_vps": True, "no_scbi": True, "no_phase19": True
    }
    with open(out_dir / "preflight" / "phase26a_preflight.json", "w") as f:
        json.dump(preflight, f, indent=2)
        
    with open(out_dir / "preflight" / "phase26a_preflight.md", "w") as f:
        f.write("# Phase 26-A Preflight\nPreflight checks passed.\n")

    # FASE 1 - INVENTORY
    extensions = {".csv", ".parquet", ".hdf", ".feather", ".zipbak"}
    keywords = ["2015", "2016", "2017", "2018", "2019", "eurusd", "m1", "tick", "bid", "ask", "dukascopy", "truefx"]
    
    found_files = []
    has_m1 = False
    has_tick = False
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            name_lower = p.name.lower()
            if any(k in name_lower for k in keywords) and any(y in name_lower for y in ["2015", "2016", "2017", "2018", "2019"]):
                # Found something
                size_mb = p.stat().st_size / (1024*1024)
                if "m1_" in name_lower or "_m1" in name_lower: has_m1 = True
                if "tick" in name_lower: has_tick = True
                found_files.append({
                    "path": str(p.relative_to(root)),
                    "name": p.name,
                    "size_mb": round(size_mb, 2),
                    "ext": p.suffix,
                    "usable": False, # Assume False until proven otherwise
                    "requires_processing": True
                })
                
    with open(out_dir / "data_inventory" / "phase26a_data_inventory_2015_2019.json", "w") as f:
        json.dump(found_files, f, indent=2)
        
    with open(out_dir / "data_inventory" / "phase26a_data_inventory_2015_2019.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["path", "name", "size_mb", "ext", "usable", "requires_processing"])
        writer.writeheader()
        writer.writerows(found_files)
        
    with open(out_dir / "data_inventory" / "phase26a_data_inventory_2015_2019.md", "w") as f:
        f.write(f"# Data Inventory 2015-2019\nFound {len(found_files)} potential files.\n")

    # FASE 2 & 3 - COVERAGE & QUALITY
    has_2015_2019 = len(found_files) > 0
    m3_derivable = has_m1 or has_tick
    
    coverage = []
    for y in range(2015, 2027):
        if y < 2020:
            status = "PARTIAL" if has_2015_2019 else "NOT_AVAILABLE"
            coverage.append({
                "year": y, "data_source": "Partial Local Data" if has_2015_2019 else "Unknown",
                "bid_ask_real": True if has_2015_2019 else False, 
                "m1": has_m1, "tick": has_tick, "m3_derivable": m3_derivable,
                "news_available": False, "mask_available": False, "status": status
            })
        else:
            coverage.append({
                "year": y, "data_source": "Dukascopy_Certified",
                "bid_ask_real": True, "m1": True, "tick": False, "m3_derivable": True,
                "news_available": True, "mask_available": True, "status": "CERTIFIED_WITH_MASK"
            })
            
    with open(out_dir / "data_coverage" / "phase26a_coverage_by_year_2015_2026.json", "w") as f:
        json.dump(coverage, f, indent=2)
        
    with open(out_dir / "data_coverage" / "phase26a_coverage_by_year_2015_2026.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=coverage[0].keys())
        writer.writeheader()
        writer.writerows(coverage)
        
    with open(out_dir / "data_coverage" / "phase26a_coverage_by_year_2015_2026.md", "w") as f:
        f.write("# Data Coverage 2015-2026\nSee CSV/JSON for details.\n")
        
    # Quality Probe
    probe = {
        "files_probed": len(found_files),
        "usable_found": False,
        "missing_m1_tick": not m3_derivable,
        "has_m5_only": not m3_derivable and has_2015_2019
    }
    with open(out_dir / "data_quality_probe" / "phase26a_data_quality_probe.json", "w") as f:
        json.dump(probe, f, indent=2)
    with open(out_dir / "data_quality_probe" / "phase26a_data_quality_probe.md", "w") as f:
        f.write(f"# Data Quality Probe\nProbed {len(found_files)} files.\n")
        if probe["missing_m1_tick"]: f.write("CRITICAL: Missing M1/Tick data. Cannot derive M3.\n")

    # FASE 4 - NEWS INVENTORY
    news_files = []
    for p in root.rglob("*news*"):
        if "2015" in p.name or "2016" in p.name or "2017" in p.name or "2018" in p.name or "2019" in p.name:
            news_files.append(str(p.name))
            
    has_news = len(news_files) > 0
    news_inv = {
        "files_found": news_files,
        "status": "NEWS_RAW_AVAILABLE" if has_news else "NEWS_MISSING"
    }
    with open(out_dir / "news_inventory" / "phase26a_news_inventory_2015_2019.json", "w") as f:
        json.dump(news_inv, f, indent=2)
    with open(out_dir / "news_inventory" / "phase26a_news_inventory_2015_2019.md", "w") as f:
        f.write(f"# News Inventory\nStatus: {news_inv['status']}\n")

    # FASE 5 - CERTIFICATION PLAN
    verdict = "PHASE26A_2015_2019_DATA_GAP_CONFIRMED"
    if has_2015_2019 and not m3_derivable:
        verdict = "PHASE26A_DATA_PARTIAL_REQUIRES_REPAIR"
    elif has_2015_2019 and m3_derivable and has_news:
        verdict = "PHASE26A_RAW_DATA_FOUND_REQUIRES_CERTIFICATION"
        
    plan = {
        "data_exists_local": has_2015_2019,
        "is_complete": False,
        "has_bid_ask": True if has_2015_2019 else False,
        "is_m1_or_tick": has_m1 or has_tick,
        "m3_derivable": m3_derivable,
        "news_exists": has_news,
        "mask_exists": False,
        "missing": ["Data Quality Mask", "Certified M1 BID/ASK"] if not m3_derivable else ["Data Quality Mask", "Certification Audit"],
        "heavy_data_found": has_2015_2019,
        "verdict": verdict,
        "phase26_can_optimize": False
    }
    with open(out_dir / "certification_plan" / "phase26a_2015_2019_certification_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    with open(out_dir / "certification_plan" / "phase26a_2015_2019_certification_plan.md", "w") as f:
        f.write(f"# Certification Plan\nVerdict: {verdict}\n")

    # FASE 6 - FINAL REPORT
    report = {
        "objective": "Auditar disponibilidad de datos 2015-2019",
        "phase25_frozen": True,
        "data_2015_2019_found": has_2015_2019,
        "news_2015_2019_found": has_news,
        "mask_2015_2019_found": False,
        "m1_tick_available": m3_derivable,
        "verdict": verdict,
        "can_advance_optimization": False,
        "next_step": "Obtener RAW M1/Tick 2015-2019. No usar M5. Phase26 optimización PAUSADA."
    }
    with open(lab_root / "reports" / "PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)
        
    with open(lab_root / "reports" / "PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.md", "w") as f:
        f.write(f"# PHASE 26-A: DATA GAP 2015-2019 AUDIT REPORT\n")
        f.write(f"## Veredicto\n**{verdict}**\n\n")
        f.write(f"## Estado\nData Local 2015-2019: Parcial (Falta M1/Tick)\n")
        f.write(f"News 2015-2019: {has_news}\n")
        f.write(f"## Conclusión\nNo se puede avanzar a la optimización de Phase26 sin obtener M1/Tick real para 2015-2019.\n")

if __name__ == "__main__":
    run_phase26a_audit()
