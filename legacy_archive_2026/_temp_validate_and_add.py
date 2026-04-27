"""
Valida e incorpora fechas CPI/PPI al manifest.
"""
import json
from pathlib import Path
from datetime import datetime

# Fechas oficiales recibidas
CPI_2024 = [
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12",
    "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11"
]

PPI_2024 = [
    "2024-01-12", "2024-02-16", "2024-03-14", "2024-04-11", "2024-05-14", "2024-06-13",
    "2024-07-12", "2024-08-13", "2024-09-12", "2024-10-11", "2024-11-14", "2024-12-12"
]

CPI_2025 = [
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
    "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24", "2025-12-18"
]

PPI_2025 = [
    "2025-01-14", "2025-02-13", "2025-03-13", "2025-04-11", "2025-05-15", "2025-06-12",
    "2025-07-16", "2025-08-14", "2025-09-10", "2025-11-25"
]

CPI_2026 = [
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10",
    "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10"
]

PPI_2026 = [
    "2026-01-14", "2026-01-30", "2026-02-27", "2026-03-18", "2026-04-14", "2026-05-13",
    "2026-06-11", "2026-07-15", "2026-08-13", "2026-09-10", "2026-10-15", "2026-11-13",
    "2026-12-15"
]

def validate_date_format(date_str):
    """Valida formato YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def check_duplicates(date_list, label):
    """Detecta duplicados en una lista."""
    seen = set()
    duplicates = []
    for d in date_list:
        if d in seen:
            duplicates.append(d)
        seen.add(d)
    return duplicates

def create_event(date_str, event_type, year):
    """Crea un evento en formato manifest."""
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

# Validación
print("=" * 60)
print("VALIDACIÓN DE FECHAS RECIBIDAS")
print("=" * 60)

all_dates = []
validation_errors = []

for year, cpi_list, ppi_list in [
    (2024, CPI_2024, PPI_2024),
    (2025, CPI_2025, PPI_2025),
    (2026, CPI_2026, PPI_2026)
]:
    print(f"\n--- {year} ---")
    
    # Validar CPI
    for d in cpi_list:
        if not validate_date_format(d):
            validation_errors.append(f"CPI {year}: formato inválido {d}")
        all_dates.append((d, "CPI", year))
    
    # Validar PPI
    for d in ppi_list:
        if not validate_date_format(d):
            validation_errors.append(f"PPI {year}: formato inválido {d}")
        all_dates.append((d, "PPI", year))
    
    # Check duplicados internos
    cpi_dups = check_duplicates(cpi_list, f"CPI {year}")
    ppi_dups = check_duplicates(ppi_list, f"PPI {year}")
    
    if cpi_dups:
        validation_errors.append(f"CPI {year}: duplicados {cpi_dups}")
    if ppi_dups:
        validation_errors.append(f"PPI {year}: duplicados {ppi_dups}")
    
    print(f"  CPI: {len(cpi_list)} fechas válidas")
    print(f"  PPI: {len(ppi_list)} fechas válidas")

# Check duplicados globales (mismo día + mismo tipo)
date_type_pairs = [(d, t) for d, t, y in all_dates]
seen_pairs = set()
duplicate_pairs = []
for pair in date_type_pairs:
    if pair in seen_pairs:
        duplicate_pairs.append(pair)
    seen_pairs.add(pair)

if duplicate_pairs:
    validation_errors.append(f"Duplicados globales: {duplicate_pairs}")

# Rareza Enero 2026 PPI
ppi_2026_jan = [d for d in PPI_2026 if d.startswith("2026-01")]
print(f"\n--- Rareza detectada ---")
print(f"PPI Enero 2026: {len(ppi_2026_jan)} fechas: {ppi_2026_jan}")

# Resumen
print(f"\n{'=' * 60}")
print("RESUMEN VALIDACIÓN")
print(f"{'=' * 60}")
print(f"Total fechas recibidas: {len(all_dates)}")
print(f"  - CPI 2024: {len(CPI_2024)}")
print(f"  - PPI 2024: {len(PPI_2024)}")
print(f"  - CPI 2025: {len(CPI_2025)}")
print(f"  - PPI 2025: {len(PPI_2025)}")
print(f"  - CPI 2026: {len(CPI_2026)}")
print(f"  - PPI 2026: {len(PPI_2026)}")

if validation_errors:
    print(f"\n❌ ERRORES DETECTADOS:")
    for e in validation_errors:
        print(f"  - {e}")
else:
    print(f"\n✓ Todas las fechas válidas. Sin duplicados. Sin errores.")

print(f"\n{'=' * 60}")
print("GENERANDO EVENTOS")
print(f"{'=' * 60}")

# Crear eventos
new_events = []
for year, cpi_list, ppi_list in [
    (2024, CPI_2024, PPI_2024),
    (2025, CPI_2025, PPI_2025),
    (2026, CPI_2026, PPI_2026)
]:
    for d in cpi_list:
        new_events.append(create_event(d, "CPI", year))
    for d in ppi_list:
        new_events.append(create_event(d, "PPI", year))

print(f"Eventos generados: {len(new_events)}")

# Leer manifest actual y agregar
manifest_path = Path('c:/Users/alera/Desktop/BOT DE TRADING CURSOR/data/official_anchors/manifests/user_curated_releases.json')
with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

original_count = len(manifest.get("releases", []))
print(f"Eventos existentes en manifest: {original_count}")

# Agregar nuevos eventos
manifest["releases"].extend(new_events)

# Guardar backup
backup_path = manifest_path.with_suffix('.json.backup')
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"Backup guardado: {backup_path}")

# Guardar manifest actualizado
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"✓ Manifest actualizado: {original_count} → {len(manifest['releases'])} eventos")
print(f"  Agregados: {len(new_events)}")
