import os
import pandas as pd
import json
from datetime import datetime, timedelta
import pytz

PROJECT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
MARKET_DATA_ROOT = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA"
TICK_PATH = os.path.join(MARKET_DATA_ROOT, "tick", "EURUSD")
HISTORICAL_LOG = os.path.join(PROJECT_ROOT, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")
DEBUG_PATH = os.path.join(PROJECT_ROOT, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical", "debug")

os.makedirs(DEBUG_PATH, exist_ok=True)

def forensic_audit_trade(trade_id, trade_info, ticks):
    """Auditoría forense detallada tick por tick."""
    direction = trade_info['type']
    sl = trade_info['sl']
    tp = trade_info['tp']
    entry_price_bar = trade_info['entry_price']
    
    risk = abs(entry_price_bar - sl)
    be_trigger = entry_price_bar + (0.4 * risk if direction == 'LONG' else -0.4 * risk)
    be_level = entry_price_bar
    
    debug_log = []
    be_active = False
    exit_reason = "NONE"
    
    for _, tick in ticks.iterrows():
        bid = tick['bid']
        ask = tick['ask']
        t_ny = tick['timestamp_ny']
        
        event = ""
        if direction == 'LONG':
            # Check BE Trigger
            if not be_active and bid >= be_trigger:
                be_active = True
                event = "BE_ACTIVATED"
            
            # Check SL/BE
            current_sl = be_level if be_active else sl
            if bid <= current_sl:
                exit_reason = "BE" if be_active else "SL"
                event = f"EXIT_{exit_reason}"
                debug_log.append({
                    "timestamp_ny": t_ny, "bid": bid, "ask": ask, 
                    "be_trigger": be_trigger, "be_active": be_active, "event": event
                })
                break
            
            # Check TP
            if bid >= tp:
                exit_reason = "TP"
                event = "EXIT_TP"
                debug_log.append({
                    "timestamp_ny": t_ny, "bid": bid, "ask": ask, 
                    "be_trigger": be_trigger, "be_active": be_active, "event": event
                })
                break
        else: # SHORT
            # Check BE Trigger
            if not be_active and ask <= be_trigger:
                be_active = True
                event = "BE_ACTIVATED"
            
            # Check SL/BE
            current_sl = be_level if be_active else sl
            if ask >= current_sl:
                exit_reason = "BE" if be_active else "SL"
                event = f"EXIT_{exit_reason}"
                debug_log.append({
                    "timestamp_ny": t_ny, "bid": bid, "ask": ask, 
                    "be_trigger": be_trigger, "be_active": be_active, "event": event
                })
                break
            
            # Check TP
            if ask <= tp:
                exit_reason = "TP"
                event = "EXIT_TP"
                debug_log.append({
                    "timestamp_ny": t_ny, "bid": bid, "ask": ask, 
                    "be_trigger": be_trigger, "be_active": be_active, "event": event
                })
                break
        
        if event:
            debug_log.append({
                "timestamp_ny": t_ny, "bid": bid, "ask": ask, 
                "be_trigger": be_trigger, "be_active": be_active, "event": event
            })

    return debug_log, exit_reason

def main():
    df_trades = pd.read_csv(HISTORICAL_LOG)
    df_jan = df_trades[df_trades['year_month'] == '2025-01'].copy()
    
    tick_file = os.path.join(TICK_PATH, "monthly", "EURUSD_ticks_2025_01.parquet")
    df_ticks = pd.read_parquet(tick_file)
    df_ticks['timestamp_ny'] = pd.to_datetime(df_ticks['timestamp_ny'])

    # Seleccionar 5 trades para debug (los primeros 5 del mes)
    sample_ids = df_jan.index[:5]
    
    for tid in sample_ids:
        trade = df_jan.loc[tid]
        t_entry = pd.to_datetime(trade['entry_time'])
        t_exit = pd.to_datetime(trade['exit_time'])
        window = df_ticks[(df_ticks['timestamp_ny'] >= t_entry) & (df_ticks['timestamp_ny'] <= t_exit + timedelta(minutes=5))]
        
        debug_log, reason = forensic_audit_trade(tid, trade, window)
        
        # Guardar CSV de debug
        debug_df = pd.DataFrame(debug_log)
        debug_df.to_csv(os.path.join(DEBUG_PATH, f"TRADE_{tid}_TICK_FORENSIC_DEBUG.csv"), index=False)
        
        # Generar MD
        md_content = f"""# FORENSIC DEBUG TRADE {tid}
- **Direction**: {trade['type']}
- **Bar Outcome**: {trade['outcome']}
- **Tick Outcome**: {reason}
- **Entry Price**: {trade['entry_price']}
- **SL**: {trade['sl']}
- **TP**: {trade['tp']}

## Event Log
{debug_df.to_string(index=False) if not debug_df.empty else "No events detected (Immediate SL?)"}
"""
        with open(os.path.join(DEBUG_PATH, f"TRADE_{tid}_TICK_FORENSIC_DEBUG.md"), "w") as f:
            f.write(md_content)
        
        print(f"Debug generated for Trade {tid}: Bar={trade['outcome']} | Tick={reason}")

if __name__ == "__main__":
    main()
