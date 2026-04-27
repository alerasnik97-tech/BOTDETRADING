"""
Valida e incorpora fechas CPI/PPI 2020-2023 al manifest.
"""
import json
from pathlib import Path
from datetime import datetime

# Fechas oficiales recibidas 2020-2023
CPI_2020 = ["2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10", "2020-05-12", "2020-06-10",
    "2020-07-14", "2020-08-12", "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10"]

PPI_2020 = ["2020-01-15", "2020-02-19", "2020-03-12", "2020-04-09", "2020-05-13", "2020-06-11",
    "2020-07-10", "2020-08-11", "2020-09-10", "2020-10-14", "2020-11-13", "2020-12-11"]

CPI_2021 = ["2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12", "2021-06-10",
    "2021-07-13", "2021-08-11", "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10"]

PPI_2021 = ["2021-01-15", "2021-02-17", "2021-03-12", "2021-04-09", "2021-05-13", "2021-06-15",
    "2021-07-14", "2021-08-12", "2021-09-10", "2021-10-14", "2021-11-09", "2021-12-14"]

CPI_2022 = ["2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11", "2022-06-10",
    "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13"]

PPI_2022 = ["2022-01-13", "2022-02-15", "2022-03-15", "2022-04-13", "2022-05-12", "2022-06-14",
    "2022-07-14", "2022-08-11", "2022-09-14", "2022-10-12", "2022-11-15", "2022-12-09"]

CPI_2023 = ["2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13",
    "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12"]

PPI_2023 = ["2023-01-18", "2023-02-16", "2023-03-15", "2023-04-13", "2023-05-11", "2023-06-14",
    "2023-07-13", "2023-08-11", "2023-09-14", "2023-10-11", "2023-11-15", "2023-12-13"]

def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def check_dups(date_list):
    seen = set()
    dups = []
    for d in date_list:
        if d in seen:
            dups.append(d)
        seen.add(d)
    return dups

def create_event(date_str, event_type, year):
    return {
        "title": f"{event_type.lower()} m/m",
        "local_date_ny": date_str,
        "local_time_ny": "08:30",
        "currency": "USD",
        "country": "United States",
        "source": f"bls_{event_type.lower()}_official_{year}",
        "source_url": f"https://www.bls.gov/schedule/{year}/home.htm",
        "anchor_group": event_type.upper(),
        "notes": f"Official BLS {event_type} release date"
    }

# Validacion
print("=" * 60)
print("VALIDACION DE FECHAS 2020-2023")
print("=" * 60)

all_dates = []
errors = []

for year, cpi_list, ppi_list in [
    (2020, CPI_2020, PPI_2020),
    (2021, CPI_2021, PPI_2021),
    (2022, CPI_2022, PPI_2022),
    (2023, CPI_2023, PPI_2023)
]:
    print(f"\n--- {year} ---")
    
    for d in cpi_list:
        if not validate_date(d):
            errors.append(f"CPI {year}: invalid {d}")
        all_dates.append((d, "CPI", year))
    
    for d in ppi_list:
        if not validate_date(d):
            errors.append(f"PPI {year}: invalid {d}")
        all_dates.append((d, "PPI", year))
    
    cpi_dups = check_dups(cpi_list)
    ppi_dups = check_dups(ppi_list)
    
    if cpi_dups:
        errors.append(f"CPI {year}: duplicates {cpi_dups}")
    if ppi_dups:
        errors.append(f"PPI {year}: duplicates {ppi_dups}")
    
    print(f"  CPI: {len(cpi_list)} fechas validas")
    print(f"  PPI: {len(ppi_list)} fechas validas")

# Check duplicados globales
date_type_pairs = [(d, t) for d, t, y in all_dates]
seen = set()
dups = []
for p in date_type_pairs:
    if p in seen:
        dups.append(p)
    seen.add(p)

if dups:
    errors.append(f"Global duplicates: {dups}")

# Resumen
print(f"\n{'=' * 60}")
print("RESUMEN")
print(f"{'=' * 60}")
print(f"Total fechas: {len(all_dates)}")
print(f"  CPI 2020: {len(CPI_2020)}")
print(f"  PPI 2020: {len(PPI_2020)}")
print(f"  CPI 2021: {len(CPI_2021)}")
print(f"  PPI 2021: {len(PPI_2021)}")
print(f"  CPI 2022: {len(CPI_2022)}")
print(f"  PPI 2022: {len(PPI_2022)}")
print(f"  CPI 2023: {len(CPI_2023)}")
print(f"  PPI 2023: {len(PPI_2023)}")

if errors:
    print(f"\n[X] ERRORES:")
    for e in errors:
        print(f"  - {e}")
else:
    print(f"\n[OK] Todas validas. Sin duplicados.")

# Crear eventos
print(f"\n{'=' * 60}")
print("GENERANDO EVENTOS")
print(f"{'=' * 60}")

new_events = []
for year, cpi_list, ppi_list in [
    (2020, CPI_2020, PPI_2020),
    (2021, CPI_2021, PPI_2021),
    (2022, CPI_2022, PPI_2022),
    (2023, CPI_2023, PPI_2023)
]:
    for d in cpi_list:
        new_events.append(create_event(d, "CPI", year))
    for d in ppi_list:
        new_events.append(create_event(d, "PPI", year))

print(f"Eventos generados: {len(new_events)}")

# Leer manifest actual
manifest_path = Path('c:/Users/alera/Desktop/BOT DE TRADING CURSOR/data/official_anchors/manifests/user_curated_releases.json')
with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

original_count = len(manifest.get("releases", []))
print(f"Eventos existentes: {original_count}")

# Agregar nuevos
manifest["releases"].extend(new_events)

# Guardar
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"[OK] Manifest actualizado: {original_count} -> {len(manifest['releases'])} eventos")
print(f"  Agregados: {len(new_events)}")
