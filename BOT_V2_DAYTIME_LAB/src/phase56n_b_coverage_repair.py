import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import pytz

# CONFIGURATION
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
REPORT_DIR = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches")
BATCH_DIR = os.path.join(REPORT_DIR, "batch_202001_202002")
RAW_TRADES_PATH = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE56_RAW_TRADES_CANONICAL.csv")
CHECKPOINT_PATH = os.path.join(REPORT_DIR, "PHASE56_FULL_HISTORICAL_CHECKPOINT.json")

# CONSTANTS
TP_R = 1.4
BE_TRIGGER_R = 0.4
COMMISSION_PIPS = 0.5 

def run_repair(year, month):
    m_str = f"{year}-{month:02d}"
    parquet_path = os.path.join(DATA_DIR, f"EURUSD_ticks_{year}_{month:02d}.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet not found {parquet_path}")
        return
        
    print(f"--- Repairing {m_str} ---")
    df_ticks = pd.read_parquet(parquet_path)
    ts_col = 'timestamp_utc' if 'timestamp_utc' in df_ticks.columns else 'timestamp'
    df_ticks['timestamp'] = pd.to_datetime(df_ticks[ts_col], utc=True)
    df_ticks = df_ticks.sort_values('timestamp')
    
    tick_times = df_ticks['timestamp'].values
    tick_bids = df_ticks['bid'].values
    tick_asks = df_ticks['ask'].values
    
    print(f"Ticks loaded: {len(df_ticks)}")
    print(f"Ticks range: {df_ticks['timestamp'].iloc[0]} to {df_ticks['timestamp'].iloc[-1]}")
    
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    # Ensure entry_time is parsed correctly with UTC
    df_raw['entry_time'] = pd.to_datetime(df_raw['entry_time'], utc=True)
    
    month_trades = df_raw[
        (df_raw['entry_time'].dt.year == year) & 
        (df_raw['entry_time'].dt.month == month)
    ].copy()
    
    print(f"Trades in sample: {len(month_trades)}")
    
    results = []
    ny_tz = pytz.timezone('America/New_York')
    
    for _, trade in month_trades.iterrows():
        entry_time = trade['entry_time'] # Already UTC Timestamp
        entry_price = trade['entry_price']
        direction = trade['type'] if 'type' in trade else trade['direction']
        
        # Reconstruct initial SL using 'risk' column (since 'sl' might be updated to entry for BE)
        risk_val = trade['risk']
        sl = entry_price - risk_val if direction == 'LONG' else entry_price + risk_val
        
        # Risk Calc
        risk_pips = abs(entry_price - sl) * 10000
        if risk_pips < 0.1: # Allow very small risk but not zero
            print(f"Trade skipped: risk_pips={risk_pips} at {entry_time}")
            continue
            
        # Proper 19:45 NY time exit
        # entry_time is UTC Timestamp. Convert to NY to find the close time.
        entry_ny = entry_time.tz_convert(ny_tz)
        exit_ny = entry_ny.replace(hour=19, minute=45, second=0, microsecond=0)
        # Handle trades that might span across days? For now keep same-day logic
        time_exit_utc = exit_ny.tz_convert('UTC')
        
        # Find start index
        start_idx = np.searchsorted(tick_times, entry_time.to_datetime64())
        end_limit_idx = np.searchsorted(tick_times, time_exit_utc.to_datetime64())
        
        window_times = tick_times[start_idx:end_limit_idx]
        window_bids = tick_bids[start_idx:end_limit_idx]
        window_asks = tick_asks[start_idx:end_limit_idx]
        
        if len(window_times) == 0:
            print(f"NO_TICKS: {entry_time} to {time_exit_utc} (Indices: {start_idx} to {end_limit_idx})")
            results.append({"verdict": "NO_AUDITABLE_NO_TICKS"})
            continue
            
        # Replay logic
        tp_price = entry_price + (TP_R * (entry_price - sl)) if direction == 'LONG' else entry_price - (TP_R * (sl - entry_price))
        be_trigger_price = entry_price + (BE_TRIGGER_R * (entry_price - sl)) if direction == 'LONG' else entry_price - (BE_TRIGGER_R * (sl - entry_price))
        
        final_verdict = "TIME_EXIT"
        exit_price = window_bids[-1] if direction == 'LONG' else window_asks[-1]
        exit_time = window_times[-1]
        be_active = False
        
        for i in range(len(window_times)):
            curr_bid = window_bids[i]
            curr_ask = window_asks[i]
            
            if direction == 'LONG':
                if not be_active and curr_bid >= be_trigger_price: be_active = True
                current_sl = entry_price if be_active else sl
                if curr_bid <= current_sl:
                    final_verdict = "BE_STOP" if be_active else "SL"
                    exit_price = current_sl
                    exit_time = window_times[i]
                    break
                if curr_bid >= tp_price:
                    final_verdict = "TP"
                    exit_price = tp_price
                    exit_time = window_times[i]
                    break
            else: # SHORT
                if not be_active and curr_ask <= be_trigger_price: be_active = True
                current_sl = entry_price if be_active else sl
                if curr_ask >= current_sl:
                    final_verdict = "BE_STOP" if be_active else "SL"
                    exit_price = current_sl
                    exit_time = window_times[i]
                    break
                if curr_ask <= tp_price:
                    final_verdict = "TP"
                    exit_price = tp_price
                    exit_time = window_times[i]
                    break
        
        gross_r = (exit_price - entry_price) / abs(entry_price - sl) if direction == 'LONG' else (entry_price - exit_price) / abs(entry_price - sl)
        comm_r = COMMISSION_PIPS / risk_pips
        net_r = gross_r - comm_r
        
        results.append({
            "month": m_str,
            "direction": direction,
            "entry_time": entry_time.isoformat(),
            "exit_time": pd.Timestamp(exit_time).isoformat(),
            "verdict": final_verdict,
            "gross_r": round(float(gross_r), 4),
            "net_r": round(float(net_r), 4),
            "comm_r": round(float(comm_r), 4)
        })

    df_res = pd.DataFrame([r for r in results if r['verdict'] != "NO_AUDITABLE_NO_TICKS"])
    auditables = len(df_res)
    total_trades = len(month_trades)
    coverage = auditables / total_trades if total_trades > 0 else 0
    print(f"Final Auditables: {auditables} / {total_trades} ({round(coverage*100,1)}%)")
    
    if auditables > 0:
        total_r_net = df_res['net_r'].sum()
        losses = len(df_res[df_res['net_r'] < 0])
        pf = abs(df_res[df_res['net_r'] > 0]['net_r'].sum() / df_res[df_res['net_r'] < 0]['net_r'].sum()) if losses > 0 else 99.0
        
        metrics = {
            "month": m_str,
            "sample_total": total_trades,
            "auditables": auditables,
            "non_auditables": total_trades - auditables,
            "total_R_net_FTMO": round(total_r_net, 4),
            "expectancy_net_FTMO": round(total_r_net / auditables, 4),
            "PF_net_FTMO": round(pf, 2)
        }
        
        # SAVE OUTPUTS
        prefix = f"PHASE56_MONTH_{year}{month:02d}"
        df_res.to_csv(os.path.join(BATCH_DIR, f"{prefix}_TRADE_LEVEL.csv"), index=False)
        with open(os.path.join(BATCH_DIR, f"{prefix}_METRICS.json"), "w") as f: json.dump(metrics, f, indent=4)
        
        # UPDATE CHECKPOINT ONLY IF COVERAGE > 80%
        if coverage >= 0.8:
            with open(CHECKPOINT_PATH, "r") as f: cp = json.load(f)
            cp["historical_progress"] = [m for m in cp["historical_progress"] if m["month"] != m_str]
            cp["historical_progress"].append({
                "month": m_str,
                "sample_total": metrics["sample_total"],
                "auditables": metrics["auditables"],
                "non_auditables": metrics["non_auditables"],
                "total_R_net_FTMO": metrics["total_R_net_FTMO"],
                "expectancy_net_FTMO": metrics["expectancy_net_FTMO"],
                "PF_net_FTMO": metrics["PF_net_FTMO"],
                "verdict_net_FTMO": "NET_FTMO_SURVIVES" if metrics["total_R_net_FTMO"] > 0 else "NET_FTMO_FAILS",
                "replay_status": "FORENSIC_COMPLETE",
                "batch_id": "PHASE56N_B_REPAIR",
                "completed_at": datetime.now().isoformat()
            })
            with open(CHECKPOINT_PATH, "w") as f: json.dump(cp, f, indent=4)
            print(f"Checkpoint updated for {m_str}")
        else:
            print(f"Coverage too low ({round(coverage*100,1)}%) for {m_str}, checkpoint NOT updated.")
    
if __name__ == "__main__":
    run_repair(2020, 1)
    run_repair(2020, 2)
