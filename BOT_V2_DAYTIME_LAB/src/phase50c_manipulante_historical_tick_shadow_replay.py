import os
import sys
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import pytz

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
MARKET_DATA_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA"
TICK_PATH = os.path.join(MARKET_DATA_ROOT, "tick", "EURUSD")
HISTORICAL_LOG = os.path.join(PROJECT_ROOT, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")

# --- TIMEZONES ---
NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

class HistoricalTickReplay:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        os.makedirs(REPORTS_PATH, exist_ok=True)

    def load_historical_trades(self, year_month=None):
        if not os.path.exists(HISTORICAL_LOG): return pd.DataFrame()
        df = pd.read_csv(HISTORICAL_LOG)
        if year_month:
            df = df[df['year_month'] == year_month].copy()
        return df

    def replay_trade(self, trade, ticks):
        """Simulación de ejecución tick-by-tick."""
        # trade: Series con entry_time, type, sl, tp, entry_price
        direction = trade['type']
        sl = trade['sl']
        tp = trade['tp']
        
        # Regla BE protegida de MANIPULANTE (0.4R)
        risk = abs(trade['entry_price'] - sl)
        be_trigger = trade['entry_price'] + (0.4 * risk if direction == 'LONG' else -0.4 * risk)
        be_level = trade['entry_price']
        
        status = "OPEN"
        exit_reason = "NONE"
        exit_time = None
        exit_price = None
        be_active = False
        
        for _, tick in ticks.iterrows():
            bid = tick['bid']
            ask = tick['ask']
            t_ny = tick['timestamp_ny']
            
            if direction == 'LONG':
                # Trigger BE
                if not be_active and bid >= be_trigger:
                    be_active = True
                
                # Check SL/BE
                current_sl = be_level if be_active else sl
                if bid <= current_sl:
                    status = "CLOSED"
                    exit_reason = "BE" if be_active else "SL"
                    exit_time = t_ny
                    exit_price = current_sl
                    break
                
                # Check TP
                if bid >= tp:
                    status = "CLOSED"
                    exit_reason = "TP"
                    exit_time = t_ny
                    exit_price = tp
                    break
            else: # SHORT
                # Trigger BE
                if not be_active and ask <= be_trigger:
                    be_active = True
                
                # Check SL/BE
                current_sl = be_level if be_active else sl
                if ask >= current_sl:
                    status = "CLOSED"
                    exit_reason = "BE" if be_active else "SL"
                    exit_time = t_ny
                    exit_price = current_sl
                    break
                
                # Check TP
                if ask <= tp:
                    status = "CLOSED"
                    exit_reason = "TP"
                    exit_time = t_ny
                    exit_price = tp
                    break
                    
        return status, exit_reason, exit_time, exit_price

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-month", type=str, default="2025-01")
    args = parser.parse_args()

    if args.dry_run: print("[DRY-RUN] Historical Replay Ready."); return

    replay = HistoricalTickReplay()
    trades = replay.load_historical_trades(args.audit_month)
    
    tick_file = os.path.join(TICK_PATH, "monthly", f"EURUSD_ticks_{args.audit_month.replace('-','_')}.parquet")
    if not os.path.exists(tick_file):
        print(f"[!] No tick data for {args.audit_month}")
        return

    print(f"[*] Auditando {len(trades)} trades de {args.audit_month}...")
    df_ticks = pd.read_parquet(tick_file)
    df_ticks['timestamp_ny'] = pd.to_datetime(df_ticks['timestamp_ny'])
    
    results = []
    for idx, trade in trades.iterrows():
        # Ventana
        t_entry = pd.to_datetime(trade['entry_time'])
        t_exit = pd.to_datetime(trade['exit_time'])
        window = df_ticks[(df_ticks['timestamp_ny'] >= t_entry) & (df_ticks['timestamp_ny'] <= t_exit + timedelta(minutes=5))]
        
        status, reason, ex_time, ex_price = replay.replay_trade(trade, window)
        
        # Comparación
        match = (reason == trade['outcome'])
        results.append({
            "trade_id": idx,
            "entry": trade['entry_time'],
            "bar_outcome": trade['outcome'],
            "tick_outcome": reason,
            "match": match,
            "bar_r": trade['r_result']
        })
        print(f"Trade {idx}: Bar={trade['outcome']} | Tick={reason} | Match={match}")

    # Reporte
    res_df = pd.DataFrame(results)
    summary_path = os.path.join(REPORTS_PATH, f"HISTORICAL_TICK_AUDIT_{args.audit_month}.csv")
    res_df.to_csv(summary_path, index=False)
    print(f"[SUCCESS] Audit completed. Match rate: {res_df['match'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()
