"""
PIPELINE DE ABASTECIMIENTO INSTITUCIONAL EURUSD (V2.2 - DUKASCOPY HARDENED)
Fuente: Dukascopy (ECN) | Calidad: Institucional | TZ: America/New_York
"""
import os
import sys
import pandas as pd
import pytz
from pathlib import Path
from datetime import datetime, timedelta

# Intentar instalar dukascopy-data
try:
    from dukascopy_data import fetch_ticks
except ImportError:
    print("[!] Libreria 'dukascopy-data' no encontrada. Instalandola...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dukascopy-data"])
        from dukascopy_data import fetch_ticks
    except:
        print("[CRITICO] No se pudo instalar el driver de Dukascopy.")
        sys.exit(1)

# RUTA CANÓNICA DE INTAKE DEL LABORATORIO
LAB_INTAKE_PATH = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data\coverage_pipeline\intake")

# TIMEZONE CANÓNICA DEL LABORATORIO
LAB_TZ = pytz.timezone('America/New_York')

# DENSIDADES REQUERIDAS
M5_REQUIRED_ROWS = 288  # 24h * 12 barras/hora
H1_REQUIRED_ROWS = 24   # 24h * 1 barra/hora

def fetch_and_process(symbol="EURUSD", days_back=7):
    print(f"=== INICIANDO ABASTECIMIENTO DUKASCOPY [{datetime.now(LAB_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}] ===")
    
    # Target date: día de operación en NY (ayer si son <5pm, hoy si son >=5pm)
    now_ny = datetime.now(LAB_TZ)
    if now_ny.hour < 17:  # Antes de cierre NY, operamos el día anterior
        target_date = (now_ny - timedelta(days=1)).date()
    else:
        target_date = now_ny.date()
    
    print(f"[*] Target date (NY): {target_date}")
    
    # Descargar suficiente para cubrir el target_date completo más margen
    end_date = datetime.utcnow() + timedelta(days=1)  # Margen hacia adelante
    start_date = end_date - timedelta(days=days_back)  # Margen hacia atrás
    
    try:
        print(f"[*] Descargando data desde {start_date.date()}...")
        # dukascopy-data suele descargar ticks
        df = fetch_ticks(symbol, start_date, end_date)
        
        if df is None or df.empty:
            print("[ERROR] No se recibieron datos.")
            return False

        # Resamplear ticks a OHLC
        df.set_index('timestamp', inplace=True)
        
        print("[*] Generando Timeframe M5...")
        df_m5 = df['ask'].resample('5min').ohlc().dropna().reset_index()
        df_m5['timestamp'] = df_m5['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LAB_TZ)
        
        print("[*] Generando Timeframe H1...")
        df_h1 = df['ask'].resample('1H').ohlc().dropna().reset_index()
        df_h1['timestamp'] = df_h1['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LAB_TZ)

        # Filtrar por target_date
        df_m5_target = df_m5[df_m5['timestamp'].dt.date == target_date].copy()
        df_h1_target = df_h1[df_h1['timestamp'].dt.date == target_date].copy()
        
        # Validaciones de densidad
        print(f"[*] Validando densidad M5: {len(df_m5_target)} filas (requerido: {M5_REQUIRED_ROWS})")
        if len(df_m5_target) != M5_REQUIRED_ROWS:
            print(f"[FAIL] M5: densidad incorrecta. Esperado {M5_REQUIRED_ROWS}, got {len(df_m5_target)}")
            print(f"[DIAG] Min timestamp M5: {df_m5['timestamp'].min()}")
            print(f"[DIAG] Max timestamp M5: {df_m5['timestamp'].max()}")
            return False
            
        print(f"[*] Validando densidad H1: {len(df_h1_target)} filas (requerido: {H1_REQUIRED_ROWS})")
        if len(df_h1_target) != H1_REQUIRED_ROWS:
            print(f"[FAIL] H1: densidad incorrecta. Esperado {H1_REQUIRED_ROWS}, got {len(df_h1_target)}")
            print(f"[DIAG] Min timestamp H1: {df_h1['timestamp'].min()}")
            print(f"[DIAG] Max timestamp H1: {df_h1['timestamp'].max()}")
            return False
        
        # Agregar timezone offset explícito para CSV
        df_m5_target['timestamp'] = df_m5_target['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df_h1_target['timestamp'] = df_h1_target['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

        cols = ['timestamp', 'open', 'high', 'low', 'close']
        for df_res, name, required in [(df_m5_target, "EURUSD_M5.csv", M5_REQUIRED_ROWS), (df_h1_target, "EURUSD_H1.csv", H1_REQUIRED_ROWS)]:
            target = LAB_INTAKE_PATH / name
            df_res[cols].to_csv(target, index=False)
            print(f"[SUCCESS] {name} inyectado: {len(df_res)} filas (target: {target_date})")

        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    fetch_and_process()
