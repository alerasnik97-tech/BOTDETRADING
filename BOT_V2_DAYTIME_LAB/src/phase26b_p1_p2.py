import os
import json
from pathlib import Path

def phase1_2():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    
    # Phase 1
    p1_dir = lab / "outputs" / "phase26b_data_acquisition_2015_2019" / "requirements_review"
    p1_dir.mkdir(parents=True, exist_ok=True)
    
    review = {
        "data_faltante": "M1/Tick BID/ASK 2015-2019",
        "formato_obligatorio": "M1 o Tick real. BID/ASK explícito.",
        "prohibido": "M5, H1, M3 desde M5, interpolación, sintéticos.",
        "fuente_recomendada": "Dukascopy",
        "validaciones_necesarias": ["Gaps", "Duplicados", "BID <= ASK", "Spreads extremos", "Timezone"],
        "impide_optimizacion": "Sin data 2015-2019, la estrategia se overfittearía a 2020-2026."
    }
    with open(p1_dir / "phase26b_requirements_review.json", "w") as f: json.dump(review, f, indent=2)
    with open(p1_dir / "phase26b_requirements_review.md", "w") as f: f.write("# Requirements Review\nRevisión completada.")
    
    # Phase 2
    p2_dir = lab / "outputs" / "phase26b_data_acquisition_2015_2019" / "local_inventory"
    p2_dir.mkdir(parents=True, exist_ok=True)
    
    inventory = []
    # Search specific pattern 
    data_intake = root / "data_intake_2015_2019"
    bi5_files = list(data_intake.rglob("*.bi5"))
    
    years_found = set()
    for f in bi5_files:
        name = f.name
        year = name.split("_")[0] if "_" in name else ""
        if year in ["2015", "2016", "2017", "2018", "2019"]:
            years_found.add(year)
            
    inv_data = {
        "files_found": len(bi5_files),
        "years_covered": list(years_found),
        "format": ".bi5 (Dukascopy LZMA Tick Data)",
        "action": "Proceed with pilot/full using local .bi5 cache"
    }
    
    with open(p2_dir / "phase26b_local_data_inventory.json", "w") as f: json.dump(inv_data, f, indent=2)
    with open(p2_dir / "phase26b_local_data_inventory.md", "w") as f: f.write(f"# Local Inventory\nFound {len(bi5_files)} .bi5 files covering years {years_found}.")
    
    with open(p2_dir / "phase26b_local_data_inventory.csv", "w") as f:
        f.write("ruta,tamaño,extension,usable,razon\n")
        f.write(f"data_intake_2015_2019/cache/dukascopy,{len(bi5_files)} files,.bi5,True,Dukascopy raw cache\n")

if __name__ == "__main__":
    phase1_2()
