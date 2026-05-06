import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# Rutas
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
TICK_DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
PHASE50M_CSV = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical", "PHASE50M_CORRECTED_TICK_TRADE_LEVEL.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")
OFFICIAL_MONTHS = ["2024-05", "2024-06", "2024-07", "2024-08", "2024-10", "2024-11", "2025-01", "2025-03", "2025-07"]

def tick_replay_model(trades_df, model='A'):
    results = []
    for month, m_trades in trades_df.groupby('month'):
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
            
            t_slice = ticks[(ticks[ts_col] >= entry_time - timedelta(minutes=5)) & (ticks[ts_col] <= exit_time + timedelta(minutes=5))]
            entry_ticks = t_slice[t_slice[ts_col] >= entry_time]
            if entry_ticks.empty: continue
            
            entry_tick = entry_ticks.iloc[0]
            
            # Niveles según Modelo
            if model == 'A':
                entry_p = trade['entry_price_bar']
                sl_p = trade['sl']
                risk = trade['risk']
            else: # B, C, D
                entry_p = entry_tick['ask'] if direction == 'LONG' else entry_tick['bid']
                if model == 'B' or model == 'D':
                    risk = trade['risk']
                    sl_p = entry_p - risk if direction == 'LONG' else entry_p + risk
                else: # Model C
                    sl_p = trade['sl']
                    risk = abs(entry_p - sl_p)

            if risk <= 0: risk = 0.0001
            tp_p = entry_p + (1.4 * risk) if direction == 'LONG' else entry_p - (1.4 * risk)
            be_trig = entry_p + (0.4 * risk) if direction == 'LONG' else entry_p - (0.4 * risk)
            be_stop = entry_p
            
            # Replay
            post_entry_ticks = entry_ticks.iloc[1:]
            tick_outcome = "FORCED_CLOSE"
            tick_R = 0
            be_active = False
            first_touch_time = None
            
            for _, t in post_entry_ticks.iterrows():
                ts = t[ts_col]
                eval_p = t['bid'] if direction == 'LONG' else t['ask']
                if not be_active:
                    if (direction == 'LONG' and eval_p >= be_trig) or (direction == 'SHORT' and eval_p <= be_trig):
                        be_active = True
                
                curr_sl = be_stop if be_active else sl_p
                tp_hit = (direction == 'LONG' and eval_p >= tp_p) or (direction == 'SHORT' and eval_p <= tp_p)
                sl_hit = (direction == 'LONG' and eval_p <= curr_sl) or (direction == 'SHORT' and eval_p >= curr_sl)
                
                if tp_hit:
                    tick_outcome = "TP"; tick_R = 1.4; first_touch_time = ts; break
                if sl_hit:
                    tick_outcome = "BE" if be_active else "SL"
                    tick_R = 0.0 if be_active else -1.0
                    first_touch_time = ts; break
                if ts >= exit_time:
                    tick_R = (eval_p - entry_p) / risk if direction == 'LONG' else (entry_p - eval_p) / risk
                    first_touch_time = ts; break

            if model == 'D': tick_R -= 0.2
            
            results.append({
                "trade_id": trade_id, "month": month, "direction": direction, "entry_time_ny": entry_time.isoformat(),
                "historical_entry": trade['entry_price_bar'], "executable_entry": entry_p,
                "entry_diff_pips": abs(trade['entry_price_bar'] - entry_p) * 10000,
                "sl": sl_p, "tp": tp_p, "be_trigger": be_trig, "be_stop": be_stop,
                "tick_outcome": tick_outcome, "tick_R": tick_R, "first_touch_time": first_touch_time.isoformat() if first_touch_time else None,
                "auditable_yes_no": "YES", "notes": f"Audit Phase 50O-B: Model {model}"
            })
    return pd.DataFrame(results)

def main():
    df_m = pd.read_csv(PHASE50M_CSV)
    
    # Generar Trade-Levels
    for mod in ['A', 'B', 'C', 'D']:
        print(f"Generando Trade-Level para Modelo {mod}...")
        res = tick_replay_model(df_m, model=mod)
        res.to_csv(os.path.join(REPORTS_DIR, f"PHASE50O_MODEL_{mod}_TRADE_LEVEL.csv"), index=False)
        
    # Comparar A vs B
    df_a = pd.read_csv(os.path.join(REPORTS_DIR, "PHASE50O_MODEL_A_TRADE_LEVEL.csv"))
    df_b = pd.read_csv(os.path.join(REPORTS_DIR, "PHASE50O_MODEL_B_TRADE_LEVEL.csv"))
    
    diff = df_a.merge(df_b, on='trade_id', suffixes=('_A', '_B'))
    diff['B_minus_A'] = diff['tick_R_B'] - diff['tick_R_A']
    diff['changed_outcome'] = diff['tick_outcome_A'] != diff['tick_outcome_B']
    
    diff.to_csv(os.path.join(REPORTS_DIR, "PHASE50O_B_MODEL_DIFF_AUDIT.csv"), index=False)
    
    print("PHASE 50O-B Auditoría Forense completada.")

if __name__ == "__main__":
    main()
