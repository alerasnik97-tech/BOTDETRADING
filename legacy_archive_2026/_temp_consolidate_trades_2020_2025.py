"""
Consolidacion de datasets de trades 2020-2025.
Une trades_realistic.csv (2022-2025) con archivos historicos 2020-2021.
"""
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Archivos fuente
SOURCE_FILES = [
    # Referencia actual (2022-2025)
    {
        'path': 'c:/Users/alera/Desktop/BOT DE TRADING CURSOR/trades_realistic.csv',
        'mapping': {
            'entry_time': 'entry_time_ny',
            'pnl': 'pnl_r',
            'result': 'result',
            'pair': 'pair',
            'exit_reason': 'exit_reason',
        },
        'name': 'trades_realistic'
    },
    # Archivos 2020-2021
    {
        'path': 'c:/Users/alera/Desktop/BOT DE TRADING CURSOR/reports_free_2020_2021_opt_adaptive_all7/20260408_082413_optimize/trades.csv',
        'mapping': {
            'entry_time': 'entry_time',  # Mismo formato, diferente nombre
            'pnl': 'pnl',  # Equivalente a pnl_r
            'result': 'result',
            'pair': 'pair',
            'exit_reason': 'exit_reason',
        },
        'name': 'free_2020_2021_opt'
    },
    {
        'path': 'c:/Users/alera/Desktop/BOT DE TRADING CURSOR/reports_free_2020_run/20260407_204320_run/trades.csv',
        'mapping': {
            'entry_time': 'entry_time',
            'pnl': 'pnl',
            'result': 'result',
            'pair': 'pair',
            'exit_reason': 'exit_reason',
        },
        'name': 'free_2020_run'
    },
]

# Schema estandarizado
STANDARD_COLUMNS = [
    'source_file',
    'pair',
    'entry_time_ny',
    'pnl_r',
    'result',
    'exit_reason',
]

def normalize_result(result_str):
    """Normaliza resultado a win/loss."""
    if not result_str:
        return 'loss'
    r = result_str.upper().strip()
    if r in ['WIN', 'W']:
        return 'win'
    elif r in ['LOSS', 'L', 'LOSE']:
        return 'loss'
    return result_str.lower()

def load_and_normalize(source):
    """Carga un archivo y normaliza al schema estandar."""
    path = Path(source['path'])
    if not path.exists():
        print(f"[SKIP] No existe: {path}")
        return []
    
    rows = []
    mapping = source['mapping']
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse entry time
                entry_time_str = row.get(mapping['entry_time'], '')
                if not entry_time_str:
                    continue
                    
                # Normalizar formato de timestamp
                dt = datetime.strptime(entry_time_str[:19], '%Y-%m-%d %H:%M:%S')
                
                # Filtrar solo 2020-2021 para archivos historicos
                # (para evitar duplicados con trades_realistic)
                if source['name'] != 'trades_realistic' and dt.year not in [2020, 2021]:
                    continue
                
                # Construir fila estandarizada
                new_row = {
                    'source_file': source['name'],
                    'pair': row.get(mapping['pair'], 'EURUSD'),
                    'entry_time_ny': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'pnl_r': float(row.get(mapping['pnl'], 0)),
                    'result': normalize_result(row.get(mapping['result'], 'loss')),
                    'exit_reason': row.get(mapping['exit_reason'], 'unknown'),
                }
                
                rows.append(new_row)
            except Exception as e:
                # Ignorar filas malformadas
                pass
    
    return rows

def main():
    print("=" * 70)
    print("CONSOLIDACION DE TRADES 2020-2025")
    print("=" * 70)
    
    all_trades = []
    stats_by_source = defaultdict(int)
    years_by_source = defaultdict(set)
    
    for source in SOURCE_FILES:
        print(f"\n[LOAD] {source['name']}")
        print(f"       Path: {source['path']}")
        
        trades = load_and_normalize(source)
        
        for t in trades:
            stats_by_source[source['name']] += 1
            years_by_source[source['name']].add(
                datetime.strptime(t['entry_time_ny'], '%Y-%m-%d %H:%M:%S').year
            )
        
        all_trades.extend(trades)
        print(f"       Loaded: {len(trades)} trades")
        if trades:
            print(f"       Years: {sorted(years_by_source[source['name']])}")
    
    # Deduplicacion por (pair, entry_time)
    seen = set()
    unique_trades = []
    duplicates = 0
    
    for trade in all_trades:
        key = (trade['pair'], trade['entry_time_ny'])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique_trades.append(trade)
    
    print(f"\n[STATS]")
    print(f"  Total trades cargados: {len(all_trades)}")
    print(f"  Duplicados detectados: {duplicates}")
    print(f"  Trades unicos: {len(unique_trades)}")
    
    # Distribucion por año
    year_counts = defaultdict(int)
    for t in unique_trades:
        year = datetime.strptime(t['entry_time_ny'], '%Y-%m-%d %H:%M:%S').year
        year_counts[year] += 1
    
    print(f"\n[DISTRIBUCION POR ANIO]")
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]} trades")
    
    # Guardar consolidado
    output_path = Path('c:/Users/alera/Desktop/BOT DE TRADING CURSOR/trades_consolidated_2020_2025.csv')
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        writer.writerows(unique_trades)
    
    print(f"\n[OUTPUT] Guardado: {output_path}")
    print(f"         Filas: {len(unique_trades)}")
    
    # Verificacion final
    print(f"\n[VERIFICACION FINAL]")
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"  Confirmado: {len(rows)} filas en archivo")
        
        years = [datetime.strptime(r['entry_time_ny'], '%Y-%m-%d %H:%M:%S').year for r in rows]
        print(f"  Años: {sorted(set(years))}")
        print(f"  Rango: {min(years)} - {max(years)}")

if __name__ == "__main__":
    main()
