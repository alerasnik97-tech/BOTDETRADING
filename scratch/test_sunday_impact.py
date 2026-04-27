import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
sys.path.append(str(PROJECT_ROOT))

from institutional_research_candidate_lab.baseline_truth_model import compute_session_levels
from institutional_research_candidate_lab.data_io import load_price_frames
from institutional_research_candidate_lab.config import default_paths

def test_sunday_impact():
    paths = default_paths(PROJECT_ROOT)
    # Una semana de ejemplo: 2026-04-13 (Lunes) a 2026-04-17 (Viernes)
    # Domingo previo: 2026-04-12
    h1, m5, coverage = load_price_frames(paths, start_date="2026-04-10", end_date="2026-04-20")
    
    print("--- Auditoria de Niveles para Lunes 2026-04-13 ---")
    levels = compute_session_levels(h1)
    
    monday_date = pd.Timestamp("2026-04-13").date()
    sunday_date = pd.Timestamp("2026-04-12").date()
    friday_date = pd.Timestamp("2026-04-10").date()
    
    if monday_date in levels:
        mon_levels = levels[monday_date]
        print(f"Lunes {monday_date}:")
        print(f"  PDH: {mon_levels['pdh']}")
        print(f"  PDL: {mon_levels['pdl']}")
        
        # Verificar que dia fue el 'previous'
        dates = sorted(pd.unique(h1.index.date))
        idx = list(dates).index(monday_date)
        prev_date = dates[idx-1]
        print(f"  Dia previo detectado: {prev_date} ({pd.Timestamp(prev_date).day_name()})")
        
        prev_bars = h1[h1.index.date == prev_date]
        print(f"  High del dia previo: {prev_bars['high'].max()}")
        print(f"  Low del dia previo: {prev_bars['low'].min()}")
        
        # Comparar con el Viernes
        fri_bars = h1[h1.index.date == friday_date]
        print(f"Viernes {friday_date}:")
        print(f"  High: {fri_bars['high'].max()}")
        print(f"  Low: {fri_bars['low'].min()}")
        
        if prev_date == sunday_date:
            print("\n[CONFIRMADO] El Lunes esta usando el Domingo como PDH/PDL.")
            if mon_levels['pdh'] != fri_bars['high'].max() or mon_levels['pdl'] != fri_bars['low'].min():
                print("[IMPACTO] El Domingo ha modificado los niveles del Lunes.")
            else:
                print("[IMPACTO] El Domingo tiene el mismo rango que el Viernes (poco probable o neutral).")
        else:
            print("\n[REFUTADO] El Lunes NO esta usando el Domingo (posiblemente porque no hay barras de domingo o el loader las quito).")
    else:
        print(f"No hay niveles calculados para el Lunes {monday_date}.")

if __name__ == "__main__":
    test_sunday_impact()
