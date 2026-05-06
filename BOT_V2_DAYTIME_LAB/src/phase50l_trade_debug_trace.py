import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz

# Rutas
TICK_DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"

def run_debug_trace(trade_info):
    print(f"\n--- DEBUG TRACE: {trade_info['entry_time']} ---")
    print(f"Tipo: {trade_info['type']}")
    
    # Cargar Ticks
    year_month = trade_info['year_month'].replace("-", "_")
    parquet_path = os.path.join(TICK_DATA_DIR, f"EURUSD_ticks_{year_month}.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"Error: No se encuentra el parquet {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    ts_col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'timestamp'
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col)
    
    # Ventana
    entry_ts = pd.to_datetime(trade_info['entry_time'], utc=True)
    exit_ts = pd.to_datetime(trade_info['exit_time'], utc=True)
    
    t_slice = df[(df[ts_col] >= entry_ts) & (df[ts_col] <= exit_ts + timedelta(minutes=5))]
    
    if t_slice.empty:
        print("Error: No hay ticks en la ventana del trade.")
        return

    # Parámetros
    entry_price = trade_info['entry_price']
    initial_sl = trade_info['initial_sl']
    tp = trade_info['tp']
    risk = trade_info['risk']
    
    # Modelo Autoridad (Phase 25/27)
    auth_be_trigger = entry_price + (0.4 * risk) if trade_info['type'] == 'LONG' else entry_price - (0.4 * risk)
    auth_be_stop = entry_price
    
    # Modelo 50K (Shadow/Erróneo)
    k50_be_trigger = entry_price + (0.7 * (tp - entry_price)) if trade_info['type'] == 'LONG' else entry_price - (0.7 * (entry_price - tp))
    k50_be_stop = entry_price + (0.4 * risk) if trade_info['type'] == 'LONG' else entry_price - (0.4 * risk)

    print(f"Entry: {entry_price:.5f}, SL: {initial_sl:.5f}, TP: {tp:.5f}, Risk: {risk:.5f}")
    print(f"Modelo Autoridad: Trigger BE @ {auth_be_trigger:.5f}, Stop BE @ {auth_be_stop:.5f}")
    print(f"Modelo 50K:       Trigger BE @ {k50_be_trigger:.5f}, Stop BE @ {k50_be_stop:.5f}")
    
    auth_active = False
    k50_active = False
    
    events = []
    
    for idx, t in t_slice.iterrows():
        bid = t['bid']
        ask = t['ask']
        ts = t[ts_col]
        
        # Simulación Autoridad
        if not auth_active:
            trigger_hit = bid >= auth_be_trigger if trade_info['type'] == 'LONG' else ask <= auth_be_trigger
            if trigger_hit:
                auth_active = True
                events.append(f"[{ts}] AUTH: BE ACTIVADO (Trigger 0.4R)")
        
        # Simulación 50K
        if not k50_active:
            trigger_hit = bid >= k50_be_trigger if trade_info['type'] == 'LONG' else ask <= k50_be_trigger
            if trigger_hit:
                k50_active = True
                events.append(f"[{ts}] 50K: BE ACTIVADO (Trigger 70% TP)")

        # Salidas Autoridad
        current_sl_auth = auth_be_stop if auth_active else initial_sl
        sl_hit_auth = bid <= current_sl_auth if trade_info['type'] == 'LONG' else ask >= current_sl_auth
        tp_hit_auth = bid >= tp if trade_info['type'] == 'LONG' else ask <= tp
        
        # Salidas 50K
        current_sl_k50 = k50_be_stop if k50_active else initial_sl
        sl_hit_k50 = bid <= current_sl_k50 if trade_info['type'] == 'LONG' else ask >= current_sl_k50
        tp_hit_k50 = bid >= tp if trade_info['type'] == 'LONG' else ask <= tp
        
        if tp_hit_auth:
            events.append(f"[{ts}] AUTH: TAKE PROFIT")
            break
        if sl_hit_auth:
            events.append(f"[{ts}] AUTH: STOP LOSS / BE EXIT")
            break
            
    print("\n--- EVENTOS AUTH ---")
    for e in [x for x in events if "AUTH" in x]:
        print(e)
        
    # Reiniciar para 50K
    events_k50 = []
    k50_active = False
    for idx, t in t_slice.iterrows():
        bid = t['bid']
        ask = t['ask']
        ts = t[ts_col]
        if not k50_active:
            trigger_hit = bid >= k50_be_trigger if trade_info['type'] == 'LONG' else ask <= k50_be_trigger
            if trigger_hit:
                k50_active = True
                events_k50.append(f"[{ts}] 50K: BE ACTIVADO (Trigger 70% TP)")
        current_sl_k50 = k50_be_stop if k50_active else initial_sl
        sl_hit_k50 = bid <= current_sl_k50 if trade_info['type'] == 'LONG' else ask >= current_sl_k50
        tp_hit_k50 = bid >= tp if trade_info['type'] == 'LONG' else ask <= tp
        if tp_hit_k50:
            events_k50.append(f"[{ts}] 50K: TAKE PROFIT")
            break
        if sl_hit_k50:
            events_k50.append(f"[{ts}] 50K: STOP LOSS / BE EXIT")
            break

    print("\n--- EVENTOS 50K ---")
    for e in events_k50:
        print(e)

if __name__ == "__main__":
    # Trade #2154
    trade_2154 = {
        'type': 'LONG',
        'entry_time': '2024-05-06 08:00:00-04:00',
        'exit_time': '2024-05-06 08:18:00-04:00',
        'year_month': '2024-05',
        'entry_price': 1.07750,
        'initial_sl': 1.07750 - 0.00038, # 1.07712
        'tp': 1.078032,
        'risk': 0.00038
    }
    run_debug_trace(trade_2154)

    # Trade #2166 (SHORT)
    trade_2166 = {
        'type': 'SHORT',
        'entry_time': '2024-05-22 13:27:00-04:00',
        'exit_time': '2024-05-22 14:00:00-04:00',
        'year_month': '2024-05',
        'entry_price': 1.08416,
        'initial_sl': 1.08416 + 0.00063, # 1.08479
        'tp': 1.083278,
        'risk': 0.00063
    }
    run_debug_trace(trade_2166)

    # Trade #2151 (SHORT - BE in Auth)
    trade_2151 = {
        'type': 'SHORT',
        'entry_time': '2024-05-01 09:15:00-04:00',
        'exit_time': '2024-05-01 09:30:00-04:00',
        'year_month': '2024-05',
        'entry_price': 1.06808,
        'initial_sl': 1.06808 + 0.00055, # 1.06863
        'tp': 1.06731,
        'risk': 0.00055
    }
    run_debug_trace(trade_2151)
