import pandas as pd
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_FILE = PROJECT_ROOT / "data" / "news_eurusd_v2_utc_audit.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_intake_2015_2019" / "news"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo de familias críticas
MAPPING = {
    "NFP": ["non-farm employment change", "nonfarm payrolls", "employment change"],
    "CPI": ["cpi m/m", "cpi y/y", "core cpi m/m", "cpi"],
    "FOMC": ["federal funds rate", "fomc statement", "fomc press conference", "fomc meeting minutes"],
    "ECB": ["main refinancing rate", "ecb press conference", "ecb monetary policy decision"]
}

def audit():
    print(f"Auditing {AUDIT_FILE}...")
    df = pd.read_csv(AUDIT_FILE, low_memory=False)
    
    # Filtrar 2015-2019
    # Usamos timestamp_utc_raw para el filtro de fecha
    df["ts_raw"] = pd.to_datetime(df["timestamp_utc_raw"], errors="coerce")
    mask = (df["ts_raw"] >= "2015-01-01") & (df["ts_raw"] <= "2019-12-31")
    hist = df[mask].copy()
    
    print(f"Found {len(hist)} rows in 2015-2019")
    
    # Normalizar nombres para búsqueda
    hist["event_lower"] = hist["raw_event_name"].str.lower().fillna("")
    
    results = []
    
    for family, aliases in MAPPING.items():
        found_mask = hist["event_lower"].apply(lambda x: any(a in x for a in aliases))
        family_df = hist[found_mask].copy()
        family_df["normalized_family"] = family
        results.append(family_df)
        print(f"Family {family}: {len(family_df)} rows found")

    # También buscar Unemployment Claims y Retail Sales para completar
    supplemental = ["unemployment claims", "retail sales"]
    found_supp = hist[hist["event_lower"].apply(lambda x: any(s in x for s in supplemental))].copy()
    found_supp["normalized_family"] = "SUPPLEMENTAL"
    results.append(found_supp)

    final_df = pd.concat(results).drop_duplicates(subset=["timestamp_utc_raw", "raw_event_name", "currency"])
    
    # Guardar candidatos
    final_df.to_csv(OUTPUT_DIR / "news_eurusd_2015_2019_fortress_candidate.csv", index=False)
    
    # Generar reporte por año
    final_df["year"] = final_df["ts_raw"].dt.year
    report = final_df.groupby(["year", "normalized_family"]).size().unstack(fill_value=0)
    report.to_csv(OUTPUT_DIR / "news_coverage_by_year_2015_2019.csv")
    
    print("\nCoverage Report:")
    print(report)
    
    # Veredicto
    # NFP: ~12 por año. CPI: ~12 por año. FOMC: ~8 por año. ECB: ~8 por año.
    # Total esperado: ~40 eventos críticos por año.
    
    return report

if __name__ == "__main__":
    report = audit()
    
    # Generar JSON de reporte
    summary = {
        "verdict": "NEWS_COVERAGE_OK" if (report >= 5).all().all() else "NEWS_COVERAGE_PARTIAL",
        "details": report.to_dict()
    }
    with open(OUTPUT_DIR / "NEWS_COVERAGE_REPAIR_REPORT_2015_2019.json", "w") as f:
        json.dump(summary, f, indent=2)
