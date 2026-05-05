import pandas as pd
import numpy as np
import os
import glob
import json
import hashlib
import argparse
from datetime import datetime, timedelta
import pytz

# Configuración de Rutas
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
TICK_DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly"
RAW_TRADES_PATH = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")
DEBUG_DIR = os.path.join(REPORTS_DIR, "debug", "phase50k")

# Meses Oficiales
OFFICIAL_MONTHS = ["2024-05", "2024-06", "2024-07", "2024-08", "2024-10", "2024-11", "2025-01", "2025-03", "2025-07"]

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_tick_data(months):
    results = []
    for month in months:
        year, mm = month.split("-")
        file_name = f"EURUSD_ticks_{year}_{mm}.parquet"
        file_path = os.path.join(TICK_DATA_DIR, file_name)
        
        status = "OK"
        notes = ""
        if not os.path.exists(file_path):
            status = "BLOCKED_MISSING_TICK_MONTH"
            results.append({"month": month, "status": status, "file": file_name})
            continue
            
        try:
            df = pd.read_parquet(file_path)
            rows = len(df)
            sha = get_sha256(file_path)
            
            # Validar timestamps (Usamos timestamp_utc)
            ts_col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'timestamp'
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            first_ts = df[ts_col].min()
            last_ts = df[ts_col].max()
            is_sorted = df[ts_col].is_monotonic_increasing
            
            # Validar precios
            bad_prices = df[df['bid'] > df['ask']]
            spread_ok = len(bad_prices) == 0
            
            results.append({
                "month": month,
                "status": status,
                "rows": rows,
                "sha256": sha,
                "first_ts": first_ts.isoformat(),
                "last_ts": last_ts.isoformat(),
                "is_sorted": is_sorted,
                "bid_le_ask": spread_ok,
                "file": file_name
            })
        except Exception as e:
            results.append({"month": month, "status": "ERROR", "notes": str(e)})
            
    df_val = pd.DataFrame(results)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df_val.to_csv(os.path.join(REPORTS_DIR, "PHASE50K_TICK_DATA_VALIDATION.csv"), index=False)
    print(f"Validación de datos tick completada. Reporte en: {REPORTS_DIR}")
    return df_val

def filter_official_trades():
    df = pd.read_csv(RAW_TRADES_PATH)
    # Renombrar columnas para consistencia interna si es necesario
    # type -> direction, year_month -> month
    df = df.rename(columns={'type': 'direction', 'year_month': 'month'})
    
    # Filtrar meses oficiales
    df_official = df[df['month'].isin(OFFICIAL_MONTHS)].copy()
    
    # Validaciones básicas
    df_official['entry_time'] = pd.to_datetime(df_official['entry_time'], utc=True)
    df_official['exit_time'] = pd.to_datetime(df_official['exit_time'], utc=True)
    
    # Guardar
    df_official.to_csv(os.path.join(REPORTS_DIR, "PHASE50K_RAW_OFFICIAL_TRADES.csv"), index=False)
    print(f"Trades oficiales filtrados: {len(df_official)}")
    return df_official

def tick_replay(trades_df, audit=False):
    reproduction_results = []
    
    # Agrupar por mes para cargar parquets una sola vez
    for month, m_trades in trades_df.groupby('month'):
        print(f"Procesando mes: {month}...")
        year, mm = month.split("-")
        parquet_path = os.path.join(TICK_DATA_DIR, f"EURUSD_ticks_{year}_{mm}.parquet")
        
        if not os.path.exists(parquet_path):
            print(f"Saltando {month}, falta Parquet.")
            continue
            
        ticks = pd.read_parquet(parquet_path)
        ts_col = 'timestamp_utc' if 'timestamp_utc' in ticks.columns else 'timestamp'
        ticks[ts_col] = pd.to_datetime(ticks[ts_col], utc=True)
        ticks = ticks.sort_values(ts_col)
        
        for idx, trade in m_trades.iterrows():
            trade_id = idx
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            direction = trade['direction']
            sl_price = trade['sl']
            tp_price = trade['tp']
            bar_outcome = trade['outcome']
            bar_R = trade['r_result']
            
            # Ventana de ticks: entry - 10m a exit + 10m
            start_window = entry_time - timedelta(minutes=10)
            end_window = exit_time + timedelta(minutes=10)
            
            t_slice = ticks[(ticks[ts_col] >= start_window) & (ticks[ts_col] <= end_window)]
            
            if t_slice.empty:
                reproduction_results.append({
                    "trade_id": trade_id, "month": month, "auditable_yes_no": "NO",
                    "non_auditable_reason": "NO_TICK_DATA", "tick_outcome": "UNKNOWN", "tick_R": 0, "match_status": "MISMATCH"
                })
                continue
            
            # Encontrar el tick de entrada real (primer tick >= entry_time)
            entry_ticks = t_slice[t_slice[ts_col] >= entry_time]
            if entry_ticks.empty:
                reproduction_results.append({
                    "trade_id": trade_id, "month": month, "auditable_yes_no": "NO",
                    "non_auditable_reason": "NO_ENTRY_TICK", "tick_outcome": "UNKNOWN", "tick_R": 0, "match_status": "MISMATCH"
                })
                continue
                
            entry_tick = entry_ticks.iloc[0]
            actual_entry_price = entry_tick['ask'] if direction == 'LONG' else entry_tick['bid']
            
            initial_risk = abs(actual_entry_price - sl_price)
            if initial_risk == 0:
                initial_risk = 0.0001
            
            be_trigger_price = actual_entry_price + (0.7 * (tp_price - actual_entry_price)) if direction == 'LONG' else actual_entry_price - (0.7 * (actual_entry_price - tp_price))
            # BE stop at +0.4R from actual entry
            be_stop_price = actual_entry_price + (0.4 * initial_risk) if direction == 'LONG' else actual_entry_price - (0.4 * initial_risk)
            
            post_entry_ticks = entry_ticks.iloc[1:]
            
            tick_outcome = "FORCED_CLOSE"
            tick_R = 0
            first_touch = "NONE"
            first_touch_time = None
            be_active = False
            
            for _, t in post_entry_ticks.iterrows():
                # Revisar salida
                if direction == 'LONG':
                    tp_hit = t['bid'] >= tp_price
                    current_sl = be_stop_price if be_active else sl_price
                    sl_hit = t['bid'] <= current_sl
                    
                    if tp_hit and sl_hit:
                        tick_outcome = "AMBIGUOUS_SAME_TIMESTAMP"
                        tick_R = -1.0
                        first_touch = "BOTH"
                        first_touch_time = t[ts_col]
                        break
                        
                    if not be_active and t['bid'] >= be_trigger_price:
                        be_active = True
                    
                    if tp_hit:
                        tick_outcome = "TP"
                        tick_R = 1.4
                        first_touch = "TP"
                        first_touch_time = t[ts_col]
                        break
                    
                    if sl_hit:
                        tick_outcome = "BE" if be_active else "SL"
                        tick_R = 0.4 if be_active else -1.0
                        first_touch = "STOP"
                        first_touch_time = t[ts_col]
                        break
                else: # SHORT
                    tp_hit = t['ask'] <= tp_price
                    current_sl = be_stop_price if be_active else sl_price
                    sl_hit = t['ask'] >= current_sl
                    
                    if tp_hit and sl_hit:
                        tick_outcome = "AMBIGUOUS_SAME_TIMESTAMP"
                        tick_R = -1.0
                        first_touch = "BOTH"
                        first_touch_time = t[ts_col]
                        break

                    if not be_active and t['ask'] <= be_trigger_price:
                        be_active = True
                    
                    if tp_hit:
                        tick_outcome = "TP"
                        tick_R = 1.4
                        first_touch = "TP"
                        first_touch_time = t[ts_col]
                        break
                    
                    if sl_hit:
                        tick_outcome = "BE" if be_active else "SL"
                        tick_R = 0.4 if be_active else -1.0
                        first_touch = "STOP"
                        first_touch_time = t[ts_col]
                        break
                
                # Cierre por tiempo si llegamos al exit_time de la barra
                if t[ts_col] >= exit_time:
                    tick_outcome = "FORCED_CLOSE"
                    if direction == 'LONG':
                        tick_R = (t['bid'] - actual_entry_price) / initial_risk
                    else:
                        tick_R = (actual_entry_price - t['ask']) / initial_risk
                    break
            
            match_status = "MATCH" if tick_outcome == bar_outcome else "MISMATCH"
            
            res = {
                "trade_id": trade_id,
                "month": month,
                "date": trade['entry_date'],
                "direction": direction,
                "entry_time_ny": entry_time.isoformat(),
                "exit_time_ny": exit_time.isoformat(),
                "entry_price_bar": trade['entry_price'],
                "entry_price_tick_side": actual_entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "be_trigger": be_trigger_price,
                "bar_outcome": bar_outcome,
                "tick_outcome": tick_outcome,
                "bar_R": bar_R,
                "tick_R": tick_R,
                "first_touch": first_touch,
                "first_touch_time": first_touch_time.isoformat() if first_touch_time else None,
                "auditable_yes_no": "YES",
                "match_status": match_status
            }
            reproduction_results.append(res)
            
            # Auditoría individual si se requiere
            if audit and len(reproduction_results) <= 10:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                with open(os.path.join(DEBUG_DIR, f"TRADE_{trade_id}_DEBUG.md"), "w") as f:
                    f.write(f"# Debug Trade {trade_id}\n")
                    f.write(f"- Entry Time: {entry_time}\n")
                    f.write(f"- Direction: {direction}\n")
                    f.write(f"- Bar Entry: {trade['entry_price']}, Tick Entry: {actual_entry_price}\n")
                    f.write(f"- SL: {sl_price}, TP: {tp_price}, BE Trig: {be_trigger_price}, BE Stop: {be_stop_price}\n")
                    f.write(f"- First Event: {first_touch} at {first_touch_time}\n")
                    f.write(f"- Tick Outcome: {tick_outcome}, Tick R: {tick_R}\n")
                    f.write(f"- Bar Outcome: {bar_outcome}, Bar R: {bar_R}\n")

    df_res = pd.DataFrame(reproduction_results)
    df_res.to_csv(os.path.join(REPORTS_DIR, "PHASE50K_INDEPENDENT_TICK_TRADE_LEVEL.csv"), index=False)
    return df_res

def calculate_metrics(df):
    # Métricas Mensuales
    monthly_metrics = []
    for month, m_df in df.groupby('month'):
        auditable = m_df[m_df['auditable_yes_no'] == 'YES']
        sample = len(m_df)
        aud_sample = len(auditable)
        
        # PF Tick
        pos_r = auditable[auditable['tick_R'] > 0]['tick_R'].sum()
        neg_r = abs(auditable[auditable['tick_R'] < 0]['tick_R'].sum())
        pf_tick = pos_r / neg_r if neg_r > 0 else np.inf
        
        # PF Bar
        pos_r_bar = m_df[m_df['bar_R'] > 0]['bar_R'].sum()
        neg_r_bar = abs(m_df[m_df['bar_R'] < 0]['bar_R'].sum())
        pf_bar = pos_r_bar / neg_r_bar if neg_r_bar > 0 else np.inf
        
        # Expectancy
        exp_tick = auditable['tick_R'].mean() if aud_sample > 0 else 0
        exp_bar = m_df['bar_R'].mean() if sample > 0 else 0
        
        # Winrate
        wr_tick = (len(auditable[auditable['tick_R'] > 0]) / aud_sample * 100) if aud_sample > 0 else 0
        wr_bar = (len(m_df[m_df['bar_R'] > 0]) / sample * 100) if sample > 0 else 0
        
        # Match Rate
        match_rate = (len(m_df[m_df['match_status'] == 'MATCH']) / sample * 100) if sample > 0 else 0
        
        monthly_metrics.append({
            "month": month,
            "sample": sample,
            "aud_sample": aud_sample,
            "pf_tick": pf_tick,
            "pf_bar": pf_bar,
            "exp_tick": exp_tick,
            "exp_bar": exp_bar,
            "wr_tick": wr_tick,
            "wr_bar": wr_bar,
            "total_r_tick": auditable['tick_R'].sum(),
            "total_r_bar": m_df['bar_R'].sum(),
            "match_rate": match_rate
        })
        
    df_monthly = pd.DataFrame(monthly_metrics)
    df_monthly.to_csv(os.path.join(REPORTS_DIR, "PHASE50K_INDEPENDENT_MONTHLY_METRICS.csv"), index=False)
    
    # Agregado
    total_auditable = df[df['auditable_yes_no'] == 'YES']
    pos_r_total = total_auditable[total_auditable['tick_R'] > 0]['tick_R'].sum()
    neg_r_total = abs(total_auditable[total_auditable['tick_R'] < 0]['tick_R'].sum())
    pf_total = pos_r_total / neg_r_total if neg_r_total > 0 else np.inf
    
    aggregate = {
        "total_trades": len(df),
        "auditables": len(total_auditable),
        "pf_tick": float(pf_total),
        "expectancy_tick": float(total_auditable['tick_R'].mean()),
        "total_r_tick": float(total_auditable['tick_R'].sum()),
        "winrate_tick": float(len(total_auditable[total_auditable['tick_R'] > 0]) / len(total_auditable) * 100) if len(total_auditable) > 0 else 0,
        "match_rate": float(len(df[df['match_status'] == 'MATCH']) / len(df) * 100),
        "best_month": df_monthly.loc[df_monthly['total_r_tick'].idxmax(), 'month'],
        "worst_month": df_monthly.loc[df_monthly['total_r_tick'].idxmin(), 'month']
    }
    
    with open(os.path.join(REPORTS_DIR, "PHASE50K_INDEPENDENT_AGGREGATE_METRICS.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
        
    return df_monthly, aggregate

def stress_tests(df):
    results = []
    
    # BASE
    base_r = df[df['auditable_yes_no'] == 'YES']['tick_R'].sum()
    results.append({"scenario": "BASE", "total_r": base_r})
    
    # EXCLUDE_NON_AUDITABLES (Ya es el base en mis cálculos)
    
    # NON_AUDITABLES_AS_SL
    df_sl = df.copy()
    df_sl.loc[df_sl['auditable_yes_no'] == 'NO', 'tick_R'] = -1.0
    results.append({"scenario": "NON_AUDITABLES_AS_SL", "total_r": df_sl['tick_R'].sum()})
    
    # EXTRA_COST_0_1R
    df_cost = df[df['auditable_yes_no'] == 'YES'].copy()
    df_cost['tick_R'] -= 0.1
    results.append({"scenario": "EXTRA_COST_0_1R", "total_r": df_cost['tick_R'].sum()})
    
    df_stress = pd.DataFrame(results)
    df_stress.to_csv(os.path.join(REPORTS_DIR, "PHASE50K_STRESS_TEST_RESULTS.csv"), index=False)
    return df_stress

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=str, default=",".join(OFFICIAL_MONTHS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    
    months = args.months.split(",")
    print(f"Iniciando Reproducción Independiente para: {months}")
    
    if args.dry_run:
        print("MODO DRY-RUN: Validando archivos solamente.")
        validate_tick_data(months)
        return

    # 1. Validar Datos
    validate_tick_data(months)
    
    # 2. Filtrar Trades
    trades = filter_official_trades()
    
    # 3. Replay
    results_df = tick_replay(trades, audit=args.audit)
    
    # 4. Métricas
    calculate_metrics(results_df)
    
    # 5. Stress Tests
    stress_tests(results_df)
    
    print("PHASE50K Completada con éxito.")

if __name__ == "__main__":
    main()
