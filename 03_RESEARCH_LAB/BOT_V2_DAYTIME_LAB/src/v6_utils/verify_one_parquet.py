
import pandas as pd
import sys
import os
from pathlib import Path

# Agregar path para v6_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_utils.temporal import sanitize_utc_index, to_ny, assert_no_dst_holes

def verify_parquet(file_path: str):
    print(f"[*] Verificando: {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"[!] Error: Archivo no encontrado.")
        return
        
    df = pd.read_parquet(file_path)
    
    # Rango inicial
    start_raw = df['timestamp_utc'].min()
    end_raw = df['timestamp_utc'].max()
    
    # Sanitizar
    df_clean = sanitize_utc_index(df)
    
    # Convertir a NY
    df_ny = to_ny(df_clean)
    
    # Verificar DST holes
    assert_no_dst_holes(df_ny)
    
    print("-" * 40)
    print(f"Rango UTC: {df_clean.index.min()} -> {df_clean.index.max()}")
    print(f"Rango NY:  {df_ny.index.min()} -> {df_ny.index.max()}")
    print(f"Ticks:     {len(df_ny)}")
    print(f"NaT/Dups:  Limpio")
    print("-" * 40)

if __name__ == "__main__":
    # Usar el más reciente como muestra
    SAMPLE = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_2026_03.parquet"
    verify_parquet(SAMPLE)
