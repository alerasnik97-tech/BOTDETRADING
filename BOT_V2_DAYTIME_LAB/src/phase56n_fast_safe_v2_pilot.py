import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import subprocess
from zoneinfo import ZoneInfo

# CONFIGURATION
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
REPORT_DIR = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches")
BATCH_DIR = os.path.join(REPORT_DIR, "batch_202001_202002")
RAW_TRADES_PATH = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv")
CHECKPOINT_PATH = os.path.join(REPORT_DIR, "PHASE56_FULL_HISTORICAL_CHECKPOINT.json")
LIVE_STATUS_PATH = os.path.join(REPORT_DIR, "PHASE56_LIVE_STATUS.txt")

# CONSTANTS
TP_R = 1.4
BE_TRIGGER_R = 0.4
COMMISSION_PIPS = 0.5 # FTMO EURUSD standard

os.makedirs(BATCH_DIR, exist_ok=True)

def update_live_status(phase, month, status_msg, parquet_ready="NO"):
    with open(LIVE_STATUS_PATH, "w") as f:
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"FASE ACTUAL: {phase}\n")
        f.write(f"MES ACTUAL: {month}\n")
        f.write(f"STATUS: {status_msg}\n")
        f.write(f"PARQUET READY: {parquet_ready}\n")

def extract_ticks(year, month):
    update_live_status("EXTRACTION", f"{year}-{month:02d}", f"Extracting ticks for {year}-{month:02d}...")
    cmd = [
        "python", 
        os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB/src/phase50s_resumable_tick_extractor.py"),
        "--year", str(year),
        "--month", str(month),
        "--symbol", "EURUSD",
        "--max-days", "31",
        "--mode", "extract"
    ]
    subprocess.run(cmd, check=True)
    
    # Finalize
    update_live_status("EXTRACTION", f"{year}-{month:02d}", f"Finalizing ticks for {year}-{month:02d}...")
    cmd_fin = ["python", os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB/src/phase50s_resumable_tick_extractor.py"), "--year", str(year), "--month", str(month), "--mode", "finalize"]
    subprocess.run(cmd_fin, check=True)
    
    # Validate
    update_live_status("EXTRACTION", f"{year}-{month:02d}", f"Validating ticks for {year}-{month:02d}...")
    cmd_val = ["python", os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB/src/phase50s_resumable_tick_extractor.py"), "--year", str(year), "--month", str(month), "--mode", "validate"]
    subprocess.run(cmd_val, check=True)

def get_parquet_path(year, month):
    return os.path.join(DATA_DIR, f"EURUSD_ticks_{year}_{month:02d}.parquet")

def run_replay(year, month):
    m_str = f"{year}-{month:02d}"
    parquet_path = get_parquet_path(year, month)
    
    if not os.path.exists(parquet_path):
        extract_ticks(year, month)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found after extraction: {parquet_path}")

    update_live_status("REPLAY_OPTIMIZED", m_str, f"Loading Parquet {m_str} and buffer...", "YES")
    df_ticks = pd.read_parquet(parquet_path)
    
    # LOAD BUFFER: For winter 19:45 NY = 00:45 UTC next day, we need next month's ticks if we are at EOM
    # To be safe, always load the next month's parquet if it exists to cover month-boundary trades
    next_month = (month % 12) + 1
    next_year = year + (1 if month == 12 else 0)
    next_parquet = get_parquet_path(next_year, next_month)
    if os.path.exists(next_parquet):
        df_next = pd.read_parquet(next_parquet)
        df_ticks = pd.concat([df_ticks, df_next], ignore_index=True)

    # Handle timestamp column name
    ts_col = 'timestamp_utc' if 'timestamp_utc' in df_ticks.columns else 'timestamp'
    if ts_col not in df_ticks.columns:
        raise KeyError(f"Neither 'timestamp_utc' nor 'timestamp' found in parquets. Columns: {df_ticks.columns.tolist()}")
        
    df_ticks['timestamp'] = pd.to_datetime(df_ticks[ts_col], utc=True)
    df_ticks = df_ticks.sort_values('timestamp') # Ensure chronological order after concat
    
    # Data Quality
    dq = {
        "rows": len(df_ticks),
        "first_ts": df_ticks['timestamp'].iloc[0].isoformat(),
        "last_ts": df_ticks['timestamp'].iloc[-1].isoformat(),
        "bid_ask_ok": bool((df_ticks['bid'] <= df_ticks['ask']).all()),
        "spread_min": float((df_ticks['ask'] - df_ticks['bid']).min()),
        "nulls": int(df_ticks.isnull().sum().sum())
    }
    
    update_live_status("REPLAY_OPTIMIZED", m_str, f"Processing trades for {m_str}...", "YES")
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw['entry_time'] = pd.to_datetime(df_raw['entry_time'], utc=True)
    
    # Filter trades for this month
    month_trades = df_raw[
        (df_raw['entry_time'].dt.year == year) & 
        (df_raw['entry_time'].dt.month == month)
    ].copy()
    
    results = []
    tick_times = df_ticks['timestamp'].values
    tick_bids = df_ticks['bid'].values
    tick_asks = df_ticks['ask'].values
    
    for _, trade in month_trades.iterrows():
        entry_price = trade['entry_price']
        # Handle direction/type
        direction = trade['type'] if 'type' in trade else trade['direction']
        
        # Reconstruct initial SL from risk column (since log might have final BE sl)
        risk_val = trade['risk']
        sl = entry_price - risk_val if direction == 'LONG' else entry_price + risk_val
        
        entry_time = trade['entry_time']
        
        # Risk Calc
        risk_pips = risk_val * 10000
        if risk_pips < 0.1: continue # Skip invalid/zero risk
        
        tp_price = entry_price + (TP_R * risk_val) if direction == 'LONG' else entry_price - (TP_R * risk_val)
        be_trigger_price = entry_price + (BE_TRIGGER_R * risk_val) if direction == 'LONG' else entry_price - (BE_TRIGGER_R * risk_val)
        
        # Time Exit 19:45 NY - Timezone Aware
        ny_tz = ZoneInfo("America/New_York")
        entry_ny = entry_time.astimezone(ny_tz)
        exit_ny = entry_ny.replace(hour=19, minute=45, second=0, microsecond=0)
        time_exit_utc = exit_ny.astimezone(ZoneInfo("UTC"))
        
        # If entry is after 19:45 NY (unlikely), set exit to next day 19:45 NY
        if time_exit_utc <= entry_time:
            exit_ny = (entry_ny + timedelta(days=1)).replace(hour=19, minute=45, second=0, microsecond=0)
            time_exit_utc = exit_ny.astimezone(ZoneInfo("UTC"))

        # Find start index
        start_idx = np.searchsorted(tick_times, entry_time.to_datetime64())
        end_limit_idx = np.searchsorted(tick_times, time_exit_utc.to_datetime64())
        
        window_times = tick_times[start_idx:end_limit_idx]
        window_bids = tick_bids[start_idx:end_limit_idx]
        window_asks = tick_asks[start_idx:end_limit_idx]
        
        if len(window_times) == 0:
            results.append({"trade_id": trade.get('trade_id', 'N/A'), "verdict": "NO_AUDITABLE_NO_TICKS"})
            continue
            
        final_verdict = "TIME_EXIT"
        exit_price = window_bids[-1] if direction == 'LONG' else window_asks[-1]
        exit_time = window_times[-1]
        be_active = False
        
        for i in range(len(window_times)):
            curr_bid = window_bids[i]
            curr_ask = window_asks[i]
            
            if direction == 'LONG':
                # Check BE trigger
                if not be_active and curr_bid >= be_trigger_price:
                    be_active = True
                
                # Check SL
                current_sl = entry_price if be_active else sl
                if curr_bid <= current_sl:
                    final_verdict = "BE_STOP" if be_active else "SL"
                    exit_price = current_sl
                    exit_time = window_times[i]
                    break
                
                # Check TP
                if curr_bid >= tp_price:
                    final_verdict = "TP"
                    exit_price = tp_price
                    exit_time = window_times[i]
                    break
            else: # SHORT
                if not be_active and curr_ask <= be_trigger_price:
                    be_active = True
                
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
        
        # Calculate R
        if direction == 'LONG':
            gross_r = (exit_price - entry_price) / abs(entry_price - sl)
        else:
            gross_r = (entry_price - exit_price) / abs(entry_price - sl)
            
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

    # Metrics
    df_res = pd.DataFrame([r for r in results if r['verdict'] != "NO_AUDITABLE_NO_TICKS"])
    if len(df_res) > 0:
        total_r_net = df_res['net_r'].sum()
        auditables = len(df_res)
        wins = len(df_res[df_res['net_r'] > 0])
        losses = len(df_res[df_res['net_r'] < 0])
        pf = abs(df_res[df_res['net_r'] > 0]['net_r'].sum() / df_res[df_res['net_r'] < 0]['net_r'].sum()) if losses > 0 else 99.0
        
        metrics = {
            "month": m_str,
            "sample_total": len(month_trades),
            "auditables": auditables,
            "non_auditables": len(month_trades) - auditables,
            "total_R_net_FTMO": round(total_r_net, 4),
            "expectancy_net_FTMO": round(total_r_net / auditables, 4) if auditables > 0 else 0,
            "PF_net_FTMO": round(pf, 2)
        }
    else:
        metrics = {"month": m_str, "sample_total": len(month_trades), "auditables": 0, "total_R_net_FTMO": 0}

    # Stress
    stress = {
        "base_net": metrics.get("total_R_net_FTMO", 0),
        "extra_01": round(metrics.get("total_R_net_FTMO", 0) - (metrics.get("auditables", 0) * 0.1), 4),
        "extra_02": round(metrics.get("total_R_net_FTMO", 0) - (metrics.get("auditables", 0) * 0.2), 4)
    }

    # SAVE OUTPUTS
    prefix = f"PHASE56_MONTH_{year}{month:02d}"
    df_res.to_csv(os.path.join(BATCH_DIR, f"{prefix}_TRADE_LEVEL.csv"), index=False)
    with open(os.path.join(BATCH_DIR, f"{prefix}_METRICS.json"), "w") as f: json.dump(metrics, f, indent=4)
    with open(os.path.join(BATCH_DIR, f"{prefix}_FTMO_NET.json"), "w") as f: json.dump({"net_r": metrics.get("total_R_net_FTMO", 0)}, f, indent=4)
    with open(os.path.join(BATCH_DIR, f"{prefix}_DATA_QUALITY.json"), "w") as f: json.dump(dq, f, indent=4)
    with open(os.path.join(BATCH_DIR, f"{prefix}_STRESS.json"), "w") as f: json.dump(stress, f, indent=4)
    
    # UPDATE CHECKPOINT
    with open(CHECKPOINT_PATH, "r") as f: cp = json.load(f)
    
    # Remove existing entry for this month if any
    cp["historical_progress"] = [m for m in cp["historical_progress"] if m["month"] != m_str]
    
    cp["historical_progress"].append({
        "month": m_str,
        "sample_total": metrics["sample_total"],
        "auditables": metrics.get("auditables", 0),
        "non_auditables": metrics.get("non_auditables", 0),
        "total_R_net_FTMO": metrics.get("total_R_net_FTMO", 0),
        "expectancy_net_FTMO": metrics.get("expectancy_net_FTMO", 0),
        "PF_net_FTMO": metrics.get("PF_net_FTMO", 0),
        "verdict_net_FTMO": "NET_FTMO_SURVIVES" if metrics.get("total_R_net_FTMO", 0) > 0 else "NET_FTMO_FAILS",
        "replay_status": "FORENSIC_COMPLETE",
        "batch_id": "PHASE56N_PILOT",
        "completed_at": datetime.now().isoformat()
    })
    
    cp["summary"]["warning"] = "GLOBAL_SUMMARY_REQUIRES_RECONCILIATION"
    
    with open(CHECKPOINT_PATH, "w") as f: json.dump(cp, f, indent=4)
    
    update_live_status("REPLAY_COMPLETED", m_str, f"Month {m_str} finished.", "YES")
    return metrics

def main():
    print("Starting Pilot Phase 56N...")
    m1 = run_replay(2020, 1)
    print(f"2020-01 Done: {m1.get('total_R_net_FTMO')} R")
    m2 = run_replay(2020, 2)
    print(f"2020-02 Done: {m2.get('total_R_net_FTMO')} R")
    print("Pilot Phase 56N Completed.")

if __name__ == "__main__":
    main()
