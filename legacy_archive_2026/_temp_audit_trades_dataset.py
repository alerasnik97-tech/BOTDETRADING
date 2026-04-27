"""
Auditoria forense del dataset de trades usado por news_impact_analysis.
"""
import csv
from datetime import datetime
from collections import Counter

TRADES_PATH = "c:/Users/alera/Desktop/BOT DE TRADING CURSOR/trades_realistic.csv"

print("=" * 70)
print("AUDITORIA FORENSE: DATASET DE TRADES")
print("=" * 70)
print(f"Archivo: {TRADES_PATH}")
print()

with open(TRADES_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f"[OK] Total filas leidas: {len(rows)}")
    print(f"[OK] Columnas detectadas: {list(rows[0].keys())}")
    print()

    # Analizar timestamps
    years = []
    timestamps = []
    parse_errors = 0
    
    for i, row in enumerate(rows):
        try:
            entry_time_str = row.get('entry_time_ny', '')
            if entry_time_str:
                dt = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
                years.append(dt.year)
                timestamps.append(dt)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:
                print(f"[ERROR] Fila {i}: entry_time_ny='{entry_time_str}' -> {e}")
    
    if parse_errors > 3:
        print(f"[ERROR] ... y {parse_errors - 3} errores mas de parseo")
    
    print()
    print("=" * 70)
    print("DISTRIBUCION POR ANIO")
    print("=" * 70)
    year_counts = Counter(years)
    for year in sorted(year_counts.keys()):
        pct = year_counts[year] / len(years) * 100
        print(f"  {year}: {year_counts[year]:>4} trades ({pct:>5.1f}%)")
    
    print()
    print("=" * 70)
    print("RANGO TEMPORAL COMPLETO")
    print("=" * 70)
    print(f"  Primer trade: {min(timestamps)}")
    print(f"  Ultimo trade: {max(timestamps)}")
    print(f"  Dias totales: {(max(timestamps) - min(timestamps)).days}")
    print(f"  Anos cubiertos: {sorted(set(years))}")
    
    print()
    print("=" * 70)
    print("VERIFICACION DE FILTRO EN news_impact_analysis_v2.py")
    print("=" * 70)
    print("  El script filtra: if entry_time.year not in [2024, 2025]")
    print("  Esto excluye automaticamente:")
    for year in sorted(year_counts.keys()):
        if year not in [2024, 2025]:
            print(f"    -> {year}: {year_counts[year]} trades EXCLUIDOS")
    
    trades_2024_2025 = sum(1 for y in years if y in [2024, 2025])
    print(f"\n  Trades que usaria el analisis (2024-2025): {trades_2024_2025}")
    print(f"  Trades disponibles en total: {len(years)}")
    print(f"  Trades ignorados: {len(years) - trades_2024_2025}")
    
    print()
    print("=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    if set(years) == {2024, 2025}:
        print("[OK] El dataset SOLO contiene trades 2024-2025")
        print("     El filtro del script es correcto pero redundante")
    elif set(years) - {2024, 2025}:
        print("[ALERTA] El dataset contiene trades de otros anos:")
        for y in sorted(set(years) - {2024, 2025}):
            print(f"         - {y}: {year_counts[y]} trades")
        print("[IMPACTO] El filtro del script los esta EXCLUYENDO del analisis")
    else:
        print("[?] Distribucion inesperada - revisar manualmente")

print()
print("=" * 70)
print("AUDITORIA COMPLETADA")
print("=" * 70)
