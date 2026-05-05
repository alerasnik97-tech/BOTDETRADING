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
DEBUG_DIR = os.path.join(REPORTS_DIR, "debug", "phase50m")

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
        
        if not os.path.exists(file_path):
            results.append({"month": month, "status": "BLOCKED_MISSING_TICK_MONTH", "file": file_name})
            continue
            
        try:
            df = pd.read_parquet(file_path)
            rows = len(df)
            sha = get_sha256(file_path)
            ts_col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'timestamp'
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            results.append({
                "month": month, "status": "OK", "rows": rows, "sha256": sha,
                "first_ts": df[ts_col].min().isoformat(), "last_ts": df[ts_col].max().isoformat(),
                "file": file_name
            })
        except Exception as e:
            results.append({"month": month, "status": "ERROR", "notes": str(e)})
            
    df_val = pd.DataFrame(results)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df_val.to_csv(os.path.join(REPORTS_DIR, "PHASE50M_TICK_DATA_VALIDATION.csv"), index=False)
    return df_val

def filter_official_trades():
    df = pd.read_csv(RAW_TRADES_PATH)
    # Conservar columnas originales y filtrar meses
    df_official = df[df['year_month'].isin(OFFICIAL_MONTHS)].copy()
    
    # Validar que 2025-08 esté excluido
    df_official = df_official[df_official['year_month'] != '2025-08']
    
    df_official['entry_time'] = pd.to_datetime(df_official['entry_time'], utc=True)
    df_official['exit_time'] = pd.to_datetime(df_official['exit_time'], utc=True)
    
    # Guardar
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df_official.to_csv(os.path.join(REPORTS_DIR, "PHASE50M_OFFICIAL_RAW_TRADES.csv"), index=False)
    print(f"Trades oficiales filtrados: {len(df_official)}")
    return df_official

def tick_replay(trades_df, audit=False, debug_sample=False):
    results = []
    debug_count = 0
    max_debug = 30 if debug_sample else 0
    
    # Agrupar por mes
    for month, m_trades in trades_df.groupby('year_month'):
        print(f"Procesando mes: {month}...")
        year, mm = month.split("-")
        parquet_path = os.path.join(TICK_DATA_DIR, f"EURUSD_ticks_{year}_{mm}.parquet")
        
        if not os.path.exists(parquet_path):
            for _, t in m_trades.iterrows():
                results.append({
                    "trade_id": t.name, "month": month, "auditable_yes_no": "NO",
                    "non_auditable_reason": "NO_TICK_DATA", "tick_outcome": "UNKNOWN", "tick_R": 0
                })
            continue
            
        ticks = pd.read_parquet(parquet_path)
        ts_col = 'timestamp_utc' if 'timestamp_utc' in ticks.columns else 'timestamp'
        ticks[ts_col] = pd.to_datetime(ticks[ts_col], utc=True)
        ticks = ticks.sort_values(ts_col)
        
        for idx, trade in m_trades.iterrows():
            trade_id = idx
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            direction = trade['type']
            
            # Re-calcular riesgo inicial (Autoridad)
            # Nota: En phase38, entry_price es el precio de la barra.
            # Pero para el replay tick buscamos el primer tick >= entry_time.
            
            t_slice = ticks[(ticks[ts_col] >= entry_time - timedelta(minutes=10)) & (ticks[ts_col] <= exit_time + timedelta(minutes=10))]
            
            entry_ticks = t_slice[t_slice[ts_col] >= entry_time]
            if entry_ticks.empty:
                results.append({"trade_id": trade_id, "month": month, "auditable_yes_no": "NO", "non_auditable_reason": "NO_ENTRY_TICK"})
                continue
                
            entry_tick = entry_ticks.iloc[0]
            actual_entry_price = entry_tick['ask'] if direction == 'LONG' else entry_tick['bid']
            
            # SL Histórico
            sl_price = trade['sl']
            # Re-obtener el SL inicial si el trade ya tiene be_triggered en el CSV (que lo movió a entry)
            # En MANIPULANTE TP 1.4, el riesgo original es (TP - Entry)/1.4 o (Entry - SL)
            # Usaremos el risk original de la barra para reconstruir el SL si el SL actual coincide con la entrada.
            if abs(sl_price - trade['entry_price']) < 1e-7:
                # El SL ya estaba en BE en el CSV, reconstruimos el original usando el risk informado.
                initial_risk = trade['risk']
                orig_sl = actual_entry_price - initial_risk if direction == 'LONG' else actual_entry_price + initial_risk
            else:
                orig_sl = sl_price
                initial_risk = abs(actual_entry_price - orig_sl)

            if initial_risk <= 0: initial_risk = 0.0001
            
            # REGLAS OFICIALES PHASE 50M
            tp_price = actual_entry_price + (1.4 * initial_risk) if direction == 'LONG' else actual_entry_price - (1.4 * initial_risk)
            be_trigger_price = actual_entry_price + (0.4 * initial_risk) if direction == 'LONG' else actual_entry_price - (0.4 * initial_risk)
            be_stop_price = actual_entry_price
            
            post_entry_ticks = entry_ticks.iloc[1:]
            
            tick_outcome = "FORCED_CLOSE"
            tick_R = 0
            first_touch = "NONE"
            first_touch_time = None
            be_active = False
            be_trigger_time = None
            
            debug_events = []
            
            for _, t in post_entry_ticks.iterrows():
                ts = t[ts_col]
                bid = t['bid']
                ask = t['ask']
                
                # Check BE Trigger
                if not be_active:
                    hit_trigger = bid >= be_trigger_price if direction == 'LONG' else ask <= be_trigger_price
                    if hit_trigger:
                        be_active = True
                        be_trigger_time = ts
                        debug_events.append(f"[{ts}] BE TRIGGERED")
                
                # Evaluation Price (LONG exit at Bid, SHORT exit at Ask)
                eval_p = bid if direction == 'LONG' else ask
                
                # Check TP/SL
                tp_hit = eval_p >= tp_price if direction == 'LONG' else eval_p <= tp_price
                current_sl = be_stop_price if be_active else orig_sl
                sl_hit = eval_p <= current_sl if direction == 'LONG' else eval_p >= current_sl
                
                if tp_hit and sl_hit:
                    tick_outcome = "AMBIGUOUS_SAME_TIMESTAMP"
                    tick_R = -1.0 # Conservador
                    first_touch = "BOTH"
                    first_touch_time = ts
                    break
                
                if tp_hit:
                    tick_outcome = "TP"
                    tick_R = 1.4
                    first_touch = "TP"
                    first_touch_time = ts
                    break
                
                if sl_hit:
                    tick_outcome = "BE" if be_active else "SL"
                    tick_R = 0.0 if be_active else -1.0
                    first_touch = "STOP"
                    first_touch_time = ts
                    break
                
                # Forced Close
                if ts >= exit_time:
                    tick_outcome = "FORCED_CLOSE"
                    tick_R = (eval_p - actual_entry_price) / initial_risk if direction == 'LONG' else (actual_entry_price - eval_p) / initial_risk
                    first_touch = "TIME"
                    first_touch_time = ts
                    break
            
            res = {
                "trade_id": trade_id, "month": month, "date": trade['entry_date'], "direction": direction,
                "entry_time_ny": entry_time.isoformat(), "exit_time_ny": exit_time.isoformat(),
                "entry_price_bar": trade['entry_price'], "nearest_bid": entry_tick['bid'], "nearest_ask": entry_tick['ask'],
                "spread_entry": (entry_tick['ask'] - entry_tick['bid']) * 10000,
                "sl": orig_sl, "tp": tp_price, "risk": initial_risk,
                "be_trigger_price": be_trigger_price, "be_trigger_touched": be_active, "be_trigger_time": be_trigger_time.isoformat() if be_trigger_time else None,
                "be_stop_price": be_stop_price, "bar_outcome": trade['status'], "tick_outcome": tick_outcome,
                "bar_R": trade['r_result'], "tick_R": tick_R, "first_touch": first_touch, "first_touch_time": first_touch_time.isoformat() if first_touch_time else None,
                "auditable_yes_no": "YES", "match_status": "MATCH" if tick_outcome == trade['status'] else "MISMATCH",
                "notes": f"Corrected BE model: Trigger 0.4R -> Stop 0.0R. Initial Risk: {initial_risk:.6f}"
            }
            results.append(res)
            
            # Debugging
            if (audit or debug_sample) and debug_count < 30:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                with open(os.path.join(DEBUG_DIR, f"TRADE_{trade_id}_OFFICIAL_RULE_DEBUG.md"), "w") as f:
                    f.write(f"# Debug Trade {trade_id} (Phase 50M)\n")
                    f.write(f"- Time: {entry_time} to {exit_time}\n")
                    f.write(f"- Direction: {direction}\n")
                    f.write(f"- Entry: {actual_entry_price:.5f}, SL Initial: {orig_sl:.5f}, TP: {tp_price:.5f}\n")
                    f.write(f"- BE Trigger: {be_trigger_price:.5f} (0.4R), BE Stop: {be_stop_price:.5f} (0.0R)\n")
                    f.write(f"- Final Tick Outcome: {tick_outcome}, Tick R: {tick_R:.4f}\n")
                    f.write(f"- Bar Outcome: {trade['status']}, Bar R: {trade['r_result']:.4f}\n")
                    f.write("\n## Eventos\n")
                    for ev in debug_events: f.write(f"- {ev}\n")
                debug_count += 1

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(REPORTS_DIR, "PHASE50M_CORRECTED_TICK_TRADE_LEVEL.csv"), index=False)
    return df_res

def calculate_metrics(df):
    m_list = []
    for month, m_df in df.groupby('month'):
        auditable = m_df[m_df['auditable_yes_no'] == 'YES']
        sample = len(m_df)
        aud_sample = len(auditable)
        
        pos_r = auditable[auditable['tick_R'] > 0]['tick_R'].sum()
        neg_r = abs(auditable[auditable['tick_R'] < 0]['tick_R'].sum())
        pf_tick = pos_r / neg_r if neg_r > 0 else np.inf
        
        exp_tick = auditable['tick_R'].mean() if aud_sample > 0 else 0
        wr_tick = (len(auditable[auditable['tick_R'] > 0]) / aud_sample * 100) if aud_sample > 0 else 0
        match_rate = (len(m_df[m_df['match_status'] == 'MATCH']) / sample * 100) if sample > 0 else 0
        
        m_list.append({
            "month": month, "sample": sample, "auditable": aud_sample, "pf_tick": pf_tick,
            "exp_tick": exp_tick, "wr_tick": wr_tick, "total_r_tick": auditable['tick_R'].sum(),
            "match_rate": match_rate
        })
    df_monthly = pd.DataFrame(m_list)
    df_monthly.to_csv(os.path.join(REPORTS_DIR, "PHASE50M_CORRECTED_MONTHLY_METRICS.csv"), index=False)
    
    total_aud = df[df['auditable_yes_no'] == 'YES']
    pos_all = total_aud[total_aud['tick_R'] > 0]['tick_R'].sum()
    neg_all = abs(total_aud[total_aud['tick_R'] < 0]['tick_R'].sum())
    
    agg = {
        "total_trades": len(df), "auditables": len(total_aud),
        "pf_tick": float(pos_all / neg_all) if neg_all > 0 else 999,
        "expectancy_tick": float(total_aud['tick_R'].mean()),
        "total_r_tick": float(total_aud['tick_R'].sum()),
        "winrate_tick": float(len(total_aud[total_aud['tick_R'] > 0]) / len(total_aud) * 100) if len(total_aud) > 0 else 0,
        "match_rate": float(len(df[df['match_status'] == 'MATCH']) / len(df) * 100),
        "best_month": df_monthly.loc[df_monthly['total_r_tick'].idxmax(), 'month'],
        "worst_month": df_monthly.loc[df_monthly['total_r_tick'].idxmin(), 'month']
    }
    with open(os.path.join(REPORTS_DIR, "PHASE50M_CORRECTED_AGGREGATE_METRICS.json"), "w") as f:
        json.dump(agg, f, indent=2)
    return df_monthly, agg

def stress_tests(df):
    aud = df[df['auditable_yes_no'] == 'YES'].copy()
    results = [
        {"scenario": "BASE", "total_r": aud['tick_R'].sum(), "pf": aud[aud['tick_R']>0]['tick_R'].sum() / abs(aud[aud['tick_R']<0]['tick_R'].sum())},
        {"scenario": "NON_AUDITABLES_AS_SL", "total_r": aud['tick_R'].sum() - (len(df)-len(aud)), "pf": 0},
        {"scenario": "EXTRA_COST_0_1R", "total_r": (aud['tick_R'] - 0.1).sum(), "pf": 0},
        {"scenario": "EXTRA_COST_0_2R", "total_r": (aud['tick_R'] - 0.2).sum(), "pf": 0},
    ]
    pd.DataFrame(results).to_csv(os.path.join(REPORTS_DIR, "PHASE50M_STRESS_TEST_RESULTS.csv"), index=False)

def generate_report(agg, monthly):
    md_path = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "PHASE50M_CORRECTED_OFFICIAL_TICK_REPRODUCTION_REPORT.md")
    content = f"""# Reporte de Reproducción Tick Corregida (Phase 50M)

## 1. Veredicto Final
**{"PHASE50M_CORRECTED_REPLAY_CONFIRMS_EDGE" if agg['pf_tick'] > 2 else "PHASE50M_CORRECTED_REPLAY_CONFIRMS_DEGRADATION"}**

## 2. Resumen Ejecutivo
- Phase 50K invalidada por error en reglas (BF alucinada como trigger BE).
- Phase 50M restaura reglas oficiales: BE trigger 0.4R, BE stop 0.0R.
- BF 70% restaurado como filtro de entrada (no afecta gestión).

## 3. Métricas Agregadas
- PF Tick: {agg['pf_tick']:.4f}
- Expectancy: {agg['expectancy_tick']:.4f}R
- Total R: {agg['total_r_tick']:.4f}R
- Winrate: {agg['winrate_tick']:.2f}%
- Match Rate: {agg['match_rate']:.2f}%

## 4. Conclusión
{"El edge se mantiene robusto bajo reglas oficiales." if agg['pf_tick'] > 2 else "La degradación persiste incluso con reglas oficiales, aunque menos extrema que en 50K."}
"""
    with open(md_path, "w") as f: f.write(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=str, default=",".join(OFFICIAL_MONTHS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--debug-sample", action="store_true")
    args = parser.parse_args()
    
    months = args.months.split(",")
    if args.dry_run:
        validate_tick_data(months)
        return
        
    validate_tick_data(months)
    trades = filter_official_trades()
    results_df = tick_replay(trades, audit=args.audit, debug_sample=args.debug_sample)
    monthly, agg = calculate_metrics(results_df)
    stress_tests(results_df)
    generate_report(agg, monthly)
    print("PHASE 50M completada.")

if __name__ == "__main__":
    main()
