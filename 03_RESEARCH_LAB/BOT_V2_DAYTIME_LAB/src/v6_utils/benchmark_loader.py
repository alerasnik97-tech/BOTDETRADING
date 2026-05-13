
import time
import os
import sys
import pandas as pd
from pathlib import Path

# Asegurar que el path incluya v6_utils
sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")

from v6_utils.memory import get_process_rss_mb, MemoryGuard, safe_collect
from v6_utils.data_loader import iter_months, load_month, iter_ticks_chunked, PARQUET_ROOT

def run_benchmark():
    start_month = "2025-01"
    end_month = "2025-12"
    months = list(iter_months(start_month, end_month))
    
    print(f"[*] Iniciando Benchmark sobre {len(months)} meses...")
    
    results = {}

    # ESCENARIO 1: BULK_LEGACY (Simulado)
    print("\n[1] Escenario: BULK_LEGACY")
    safe_collect()
    rss_init = get_process_rss_mb()
    start_t = time.time()
    try:
        with MemoryGuard(budget_mb=16384, label="BULK_LEGACY", abort_on_breach=False) as guard:
            all_dfs = []
            for y, m in months:
                # Cargar todo sin filtrar columnas ni downcast (simulado cargando todas las disponibles)
                df = pd.read_parquet(PARQUET_ROOT / f"EURUSD_ticks_{y}_{m:02d}.parquet")
                all_dfs.append(df)
                guard.check()
            
            full_data = pd.concat(all_dfs)
            results["BULK_LEGACY"] = {
                "peak_rss": guard.peak_rss,
                "time": time.time() - start_t,
                "final_rss": get_process_rss_mb()
            }
            del full_data
            del all_dfs
    except Exception as e:
        print(f"Error en BULK_LEGACY: {e}")
    
    safe_collect()

    # ESCENARIO 2: BULK_SELECTIVE
    print("\n[2] Escenario: BULK_SELECTIVE")
    rss_init = get_process_rss_mb()
    start_t = time.time()
    try:
        with MemoryGuard(budget_mb=16384, label="BULK_SELECTIVE") as guard:
            all_dfs = []
            for y, m in months:
                df = load_month(y, m, columns=["timestamp_utc", "bid", "ask"], downcast_floats=True)
                all_dfs.append(df)
                guard.check()
            
            full_data = pd.concat(all_dfs)
            results["BULK_SELECTIVE"] = {
                "peak_rss": guard.peak_rss,
                "time": time.time() - start_t,
                "final_rss": get_process_rss_mb()
            }
            del full_data
            del all_dfs
    except Exception as e:
        print(f"Error en BULK_SELECTIVE: {e}")
        
    safe_collect()

    # ESCENARIO 3: STREAM_CHUNKED
    print("\n[3] Escenario: STREAM_CHUNKED")
    rss_init = get_process_rss_mb()
    start_t = time.time()
    try:
        with MemoryGuard(budget_mb=4096, label="STREAM_CHUNKED") as guard:
            count = 0
            for df in iter_ticks_chunked(start_month, end_month, columns=["timestamp_utc", "bid", "ask"]):
                # Procesamiento dummy
                _ = df["bid"].mean()
                count += len(df)
                guard.check()
                del df # Crítico para el stream
                
            results["STREAM_CHUNKED"] = {
                "peak_rss": guard.peak_rss,
                "time": time.time() - start_t,
                "final_rss": get_process_rss_mb()
            }
    except Exception as e:
        print(f"Error en STREAM_CHUNKED: {e}")

    safe_collect()
    
    # Reporte Final
    print("\n" + "="*50)
    print("RESULTADOS DEL BENCHMARK")
    print("="*50)
    for k, v in results.items():
        print(f"{k}:")
        print(f"  Peak RAM:  {v['peak_rss']:.2f} MB")
        print(f"  Tiempo:    {v['time']:.2f} s")
        print(f"  Final RAM: {v['final_rss']:.2f} MB")
    
    # Guardar en CSV/JSON
    df_res = pd.DataFrame(results).T
    df_res.to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\v6_02_memory_chunks\V6_02_BENCHMARK.csv")
    df_res.to_json(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\v6_02_memory_chunks\V6_02_BENCHMARK.json")

if __name__ == "__main__":
    run_benchmark()
