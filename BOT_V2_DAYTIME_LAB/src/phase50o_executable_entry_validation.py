import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime, timedelta

# Configuración de Rutas
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
TICK_DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
PHASE50M_CSV = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical", "PHASE50M_CORRECTED_TICK_TRADE_LEVEL.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")
DEBUG_DIR = os.path.join(REPORTS_DIR, "debug", "phase50o")

OFFICIAL_MONTHS = ["2024-05", "2024-06", "2024-07", "2024-08", "2024-10", "2024-11", "2025-01", "2025-03", "2025-07"]

def tick_replay_executable(trades_df, model='B', audit=False):
    results = []
    
    # Agrupar por mes
    for month, m_trades in trades_df.groupby('month'):
        print(f"Auditando {month} bajo Modelo {model}...")
        year, mm = month.split("-")
        parquet_path = os.path.join(TICK_DATA_DIR, f"EURUSD_ticks_{year}_{mm}.parquet")
        
        if not os.path.exists(parquet_path): continue
            
        ticks = pd.read_parquet(parquet_path)
        ts_col = 'timestamp_utc' if 'timestamp_utc' in ticks.columns else 'timestamp'
        ticks[ts_col] = pd.to_datetime(ticks[ts_col], utc=True)
        ticks = ticks.sort_values(ts_col)
        
        for idx, trade in m_trades.iterrows():
            if trade['auditable_yes_no'] == 'NO': continue
            
            trade_id = trade['trade_id']
            entry_time = pd.to_datetime(trade['entry_time_ny'], utc=True)
            exit_time = pd.to_datetime(trade['exit_time_ny'], utc=True)
            direction = trade['direction']
            
            # 1. Entrada Ejecutable
            t_slice = ticks[(ticks[ts_col] >= entry_time - timedelta(minutes=5)) & (ticks[ts_col] <= exit_time + timedelta(minutes=5))]
            entry_ticks = t_slice[t_slice[ts_col] >= entry_time]
            if entry_ticks.empty: continue
            
            entry_tick = entry_ticks.iloc[0]
            exec_entry = entry_tick['ask'] if direction == 'LONG' else entry_tick['bid']
            
            # 2. Recalcular Niveles
            if model == 'B':
                # Preserve Historical Risk Distance
                risk = trade['risk']
                exec_sl = exec_entry - risk if direction == 'LONG' else exec_entry + risk
            elif model == 'C':
                # Anchor to Historical SL
                exec_sl = trade['sl']
                risk = abs(exec_entry - exec_sl)
            else: # Model A (Historical baseline)
                exec_entry = trade['entry_price_bar']
                exec_sl = trade['sl']
                risk = trade['risk']

            if risk <= 0: risk = 0.0001
            
            exec_tp = exec_entry + (1.4 * risk) if direction == 'LONG' else exec_entry - (1.4 * risk)
            exec_be_trig = exec_entry + (0.4 * risk) if direction == 'LONG' else exec_entry - (0.4 * risk)
            exec_be_stop = exec_entry
            
            # 3. Replay
            post_entry_ticks = entry_ticks.iloc[1:]
            tick_outcome = "FORCED_CLOSE"
            tick_R = 0
            be_active = False
            first_touch_time = None
            
            for _, t in post_entry_ticks.iterrows():
                ts = t[ts_col]
                eval_p = t['bid'] if direction == 'LONG' else t['ask']
                
                if not be_active:
                    hit_trig = eval_p >= exec_be_trig if direction == 'LONG' else eval_p <= exec_be_trig
                    if hit_trig: be_active = True
                
                curr_sl = exec_be_stop if be_active else exec_sl
                tp_hit = eval_p >= exec_tp if direction == 'LONG' else eval_p <= exec_tp
                sl_hit = eval_p <= curr_sl if direction == 'LONG' else eval_p >= curr_sl
                
                if tp_hit and sl_hit:
                    tick_outcome = "AMBIGUOUS"; tick_R = -1.0; first_touch_time = ts; break
                if tp_hit:
                    tick_outcome = "TP"; tick_R = 1.4; first_touch_time = ts; break
                if sl_hit:
                    tick_outcome = "BE" if be_active else "SL"
                    tick_R = 0.0 if be_active else -1.0
                    first_touch_time = ts; break
                if ts >= exit_time:
                    tick_R = (eval_p - exec_entry) / risk if direction == 'LONG' else (exec_entry - eval_p) / risk
                    first_touch_time = ts; break

            results.append({
                "trade_id": trade_id, "month": month, "model": model, "tick_outcome": tick_outcome, "tick_R": tick_R,
                "entry_exec": exec_entry, "sl_exec": exec_sl, "tp_exec": exec_tp, "risk_exec": risk,
                "be_trig_exec": exec_be_trig, "first_touch_time": first_touch_time
            })
            
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--months", type=str, default=",".join(OFFICIAL_MONTHS))
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry-run OK")
        return
        
    df_m = pd.read_csv(PHASE50M_CSV)
    
    # Replay Modelos
    res_a = tick_replay_executable(df_m, model='A')
    res_b = tick_replay_executable(df_m, model='B')
    res_c = tick_replay_executable(df_m, model='C')
    
    all_res = pd.concat([res_a, res_b, res_c])
    
    # Métricas Comparativas
    comparison = []
    for mod in ['A', 'B', 'C']:
        m_df = all_res[all_res['model'] == mod]
        pos_r = m_df[m_df['tick_R'] > 0]['tick_R'].sum()
        neg_r = abs(m_df[m_df['tick_R'] < 0]['tick_R'].sum())
        pf = pos_r / neg_r if neg_r > 0 else 999
        comparison.append({
            "model": mod, "trades": len(m_df), "pf": pf, "expectancy": m_df['tick_R'].mean(),
            "total_r": m_df['tick_R'].sum(), "winrate": (len(m_df[m_df['tick_R']>0])/len(m_df))*100 if len(m_df)>0 else 0
        })
    
    # Model D (Model B + Extra Cost)
    base_b = comparison[1].copy()
    for cost in [0.1, 0.2, 0.3]:
        d_df = res_b.copy()
        d_df['tick_R'] -= cost
        pos_d = d_df[d_df['tick_R'] > 0]['tick_R'].sum()
        neg_d = abs(d_df[d_df['tick_R'] < 0]['tick_R'].sum())
        comparison.append({
            "model": f"D_{cost}R", "trades": len(d_df), "pf": pos_d / neg_d if neg_d > 0 else 999,
            "expectancy": d_df['tick_R'].mean(), "total_r": d_df['tick_R'].sum(),
            "winrate": (len(d_df[d_df['tick_R']>0])/len(d_df))*100 if len(d_df)>0 else 0
        })
        
    df_comp = pd.DataFrame(comparison)
    df_comp.to_csv(os.path.join(REPORTS_DIR, "PHASE50O_EXECUTABLE_MODEL_COMPARISON.csv"), index=False)
    
    # Severe Audit
    df_n = pd.read_csv(os.path.join(REPORTS_DIR, "PHASE50N_ENTRY_PRICE_REALISM_AUDIT.csv"))
    severe_ids = df_n[df_n['diff_pips'] > 7]['trade_id'].tolist()
    severe_audit = res_b[res_b['trade_id'].isin(severe_ids)].copy()
    severe_audit.to_csv(os.path.join(REPORTS_DIR, "PHASE50O_SEVERE_ENTRY_TRADES.csv"), index=False)
    
    print("PHASE 50O completada.")

if __name__ == "__main__":
    main()
