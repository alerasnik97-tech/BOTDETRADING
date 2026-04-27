"""
PIPELINE DE CONTINGENCIA EURUSD (V2.0 - YFINANCE HARDENED)
Fuente: Yahoo Finance | TZ: America/New_York | Validación: 288/24 filas
"""
import os
import pandas as pd
import pytz
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# RUTA CANÓNICA DE INTAKE DEL LABORATORIO
LAB_INTAKE_PATH = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data\coverage_pipeline\intake")

# TIMEZONE CANÓNICA DEL LABORATORIO
LAB_TZ = pytz.timezone('America/New_York')

# DENSIDADES REQUERIDAS
M5_REQUIRED_ROWS = 288  # 24h * 12 barras/hora
H1_REQUIRED_ROWS = 24     # 24h * 1 barra/hora

def fetch_contingency_data(symbol="EURUSD=X", days_back=7):
    print(f"=== INICIANDO ABASTECIMIENTO YFINANCE [{datetime.now(LAB_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}] ===")
    
    if not LAB_INTAKE_PATH.exists():
        LAB_INTAKE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Target date: día de operación en NY (ayer, datos completos garantizados)
    now_ny = datetime.now(LAB_TZ)
    target_date = (now_ny - timedelta(days=1)).date()
    
    print(f"[*] Target date (NY): {target_date}")

    try:
        # 1. Descarga H1
        print(f"[*] Descargando H1 (Intervalo: 1h, Dias: {days_back})...")
        df_h1 = yf.download(symbol, period=f"{days_back}d", interval="1h", progress=False)
        
        # 2. Descarga M5
        print(f"[*] Descargando M5 (Intervalo: 5m, Dias: {days_back})...")
        df_m5 = yf.download(symbol, period=f"{days_back}d", interval="5m", progress=False)

        if df_h1.empty or df_m5.empty:
            print("[ERROR] No se recibieron datos de Yahoo Finance.")
            return False
        
        print(f"[*] Datos crudos H1: {len(df_h1)} filas ({df_h1.index.min()} a {df_h1.index.max()})")
        print(f"[*] Datos crudos M5: {len(df_m5)} filas ({df_m5.index.min()} a {df_m5.index.max()})")

        # 3. Procesamiento Común
        datasets_raw = [
            (df_h1, "EURUSD_H1.csv", H1_REQUIRED_ROWS),
            (df_m5, "EURUSD_M5.csv", M5_REQUIRED_ROWS)
        ]
        
        processed = {}

        for df_raw, filename, required_rows in datasets_raw:
            # Normalizacion
            df = df_raw.reset_index().copy()
            
            # Aplanar MultiIndex de columnas si existe
            if isinstance(df.columns, pd.MultiIndex):
                # Tomar solo el primer nivel del MultiIndex
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Mapeo de columnas simple (ahora son strings)
            mapping = {}
            for col in df.columns:
                col_name = str(col).lower()
                if 'date' in col_name or col_name == 'datetime':
                    mapping[col] = 'timestamp'
                elif col_name == 'open':
                    mapping[col] = 'open'
                elif col_name == 'high':
                    mapping[col] = 'high'
                elif col_name == 'low':
                    mapping[col] = 'low'
                elif col_name == 'close' and 'adj' not in col_name:
                    mapping[col] = 'close'
                elif col_name == 'volume':
                    mapping[col] = 'volume'
            
            df = df[list(mapping.keys())].rename(columns=mapping)
            
            # Limpieza y timezone
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(LAB_TZ)
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            df = df.dropna()
            
            # Filtrar por target_date
            df_target = df[df['timestamp'].dt.date == target_date].copy()
            
            print(f"[*] {filename}: {len(df_target)} filas para target {target_date} (requerido: {required_rows})")
            
            # Validación de densidad
            if len(df_target) != required_rows:
                print(f"[FAIL] {filename}: densidad incorrecta. Esperado {required_rows}, got {len(df_target)}")
                print(f"[DIAG] Min timestamp disponible: {df['timestamp'].min()}")
                print(f"[DIAG] Max timestamp disponible: {df['timestamp'].max()}")
                print(f"[DIAG] Fechas disponibles: {sorted(df['timestamp'].dt.date.unique())}")
                return False
            
            # Guardado
            df_target['timestamp'] = df_target['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
            target = LAB_INTAKE_PATH / filename
            df_target[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_csv(target, index=False)
            print(f"[SUCCESS] {filename} inyectado: {len(df_target)} filas (target: {target_date})")
            processed[filename] = len(df_target)

        return True

    except Exception as e:
        print(f"[ERROR] Critico en contingencia: {e}")
        return False

if __name__ == "__main__":
    fetch_contingency_data(days_back=7)
