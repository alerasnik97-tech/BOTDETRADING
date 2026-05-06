import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import subprocess
from zoneinfo import ZoneInfo
import time

# CONFIGURATION
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
OUTPUT_ROOT = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56o_corrected_full")
RAW_TRADES_PATH = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv")
CHECKPOINT_PATH = os.path.join(OUTPUT_ROOT, "PHASE56O_CORRECTED_FULL_CHECKPOINT.json")
MANIFEST_PATH = os.path.join(OUTPUT_ROOT, "PHASE56O_PARQUET_MANIFEST.json")
LIVE_STATUS_PATH = os.path.join(OUTPUT_ROOT, "PHASE56O_LIVE_STATUS.txt")

# CONSTANTS
TP_R = 1.4
BE_TRIGGER_R = 0.4
COMMISSION_PIPS = 0.5 # FTMO EURUSD standard
SLIPPAGE_PIPS = 0.40 # From Phase 61 friction model
SPREAD_PIPS = 0.30 # Average spread estimate
ENGINE_VERSION = "PHASE56O_CORRECTED_TIMEZONE_SL_BUFFER_ENGINE"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def update_live_status(phase, month, status_msg, trades_done=0, trades_total=0):
    with open(LIVE_STATUS_PATH, "w") as f:
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"FASE ACTUAL: {phase}\n")
        f.write(f"MES ACTUAL: {month}\n")
        f.write(f"STATUS: {status_msg}\n")
        f.write(f"PROGRESS: {trades_done}/{trades_total}\n")

def get_parquet_path(year, month):
    return os.path.join(DATA_DIR, f"EURUSD_ticks_{year}_{month:02d}.parquet")

def validate_parquet(path):
    if not os.path.exists(path):
        return False, "MISSING"
    try:
        # Lightweight check
        df_head = pd.read_parquet(path, columns=['timestamp_utc' if 'timestamp_utc' in pd.read_parquet(path, engine='pyarrow').columns else 'timestamp', 'bid', 'ask']).head(10)
        if len(df_head) > 0:
            return True, "VALID"
        return False, "EMPTY"
    except Exception as e:
        return False, f"ERROR: {str(e)}"

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

def run_replay_month(year, month, df_raw_all):
    m_str = f"{year}-{month:02d}"
    month_dir = os.path.join(OUTPUT_ROOT, f"month_{year}{month:02d}")
    os.makedirs(month_dir, exist_ok=True)
    
    parquet_path = get_parquet_path(year, month)
    
    if not os.path.exists(parquet_path):
        extract_ticks(year, month)
        if not os.path.exists(parquet_path):
            return {"month": m_str, "status": "MISSING_PARQUET"}

    update_live_status("REPLAY", m_str, "Loading Parquet and buffer...")
    
    try:
        df_ticks = pd.read_parquet(parquet_path)
        
        # Load buffer for EOM trades (Winter NY exit = 00:45 UTC next day)
        next_month = (month % 12) + 1
        next_year = year + (1 if month == 12 else 0)
        next_parquet = get_parquet_path(next_year, next_month)
        if os.path.exists(next_parquet):
            df_next = pd.read_parquet(next_parquet)
            df_ticks = pd.concat([df_ticks, df_next], ignore_index=True)
            
        ts_col = 'timestamp_utc' if 'timestamp_utc' in df_ticks.columns else 'timestamp'
        df_ticks['timestamp'] = pd.to_datetime(df_ticks[ts_col], utc=True)
        df_ticks = df_ticks.sort_values('timestamp')
        
        tick_times = df_ticks['timestamp'].values
        tick_bids = df_ticks['bid'].values
        tick_asks = df_ticks['ask'].values
        
        dq = {
            "rows": len(df_ticks),
            "first_ts": df_ticks['timestamp'].iloc[0].isoformat(),
            "last_ts": df_ticks['timestamp'].iloc[-1].isoformat(),
            "bid_ask_ok": bool((df_ticks['bid'] <= df_ticks['ask']).all()),
            "spread_min": float((df_ticks['ask'] - df_ticks['bid']).min())
        }
    except Exception as e:
        return {"month": m_str, "status": f"PARQUET_ERROR: {str(e)}"}

    # Filter trades
    month_trades = df_raw_all[
        (df_raw_all['entry_time_utc'].dt.year == year) & 
        (df_raw_all['entry_time_utc'].dt.month == month)
    ].copy()
    
    if len(month_trades) == 0:
        return {"month": m_str, "status": "NO_TRADES"}

    results = []
    ny_tz = ZoneInfo("America/New_York")
    
    for idx, trade in month_trades.iterrows():
        entry_price = trade['entry_price']
        direction = trade['type'] if 'type' in trade else trade['direction']
        risk_val = trade['risk']
        sl = entry_price - risk_val if direction == 'LONG' else entry_price + risk_val
        entry_time = trade['entry_time_utc']
        
        risk_pips = risk_val * 10000
        if risk_pips < 0.1: continue
        
        tp_price = entry_price + (TP_R * risk_val) if direction == 'LONG' else entry_price - (TP_R * risk_val)
        be_trigger_price = entry_price + (BE_TRIGGER_R * risk_val) if direction == 'LONG' else entry_price - (BE_TRIGGER_R * risk_val)
        
        # NY Exit Calculation
        entry_ny = entry_time.astimezone(ny_tz)
        exit_ny = entry_ny.replace(hour=19, minute=45, second=0, microsecond=0)
        time_exit_utc = exit_ny.astimezone(ZoneInfo("UTC"))
        
        if time_exit_utc <= entry_time:
            exit_ny = (entry_ny + timedelta(days=1)).replace(hour=19, minute=45, second=0, microsecond=0)
            time_exit_utc = exit_ny.astimezone(ZoneInfo("UTC"))

        start_idx = np.searchsorted(tick_times, entry_time.to_datetime64())
        end_limit_idx = np.searchsorted(tick_times, time_exit_utc.to_datetime64())
        
        w_times = tick_times[start_idx:end_limit_idx]
        w_bids = tick_bids[start_idx:end_limit_idx]
        w_asks = tick_asks[start_idx:end_limit_idx]
        
        if len(w_times) == 0:
            results.append({"trade_id": idx, "verdict": "NO_AUDITABLE_NO_TICKS"})
            continue
            
        final_verdict = "TIME_EXIT"
        exit_price = w_bids[-1] if direction == 'LONG' else w_asks[-1]
        exit_time = w_times[-1]
        be_active = False
        
        for i in range(len(w_times)):
            c_bid = w_bids[i]
            c_ask = w_asks[i]
            
            if direction == 'LONG':
                if not be_active and c_bid >= be_trigger_price: be_active = True
                curr_sl = entry_price if be_active else sl
                if c_bid <= curr_sl:
                    final_verdict = "BE_STOP" if be_active else "SL"
                    exit_price, exit_time = curr_sl, w_times[i]
                    break
                if c_bid >= tp_price:
                    final_verdict = "TP"
                    exit_price, exit_time = tp_price, w_times[i]
                    break
            else: # SHORT
                if not be_active and c_ask <= be_trigger_price: be_active = True
                curr_sl = entry_price if be_active else sl
                if c_ask >= curr_sl:
                    final_verdict = "BE_STOP" if be_active else "SL"
                    exit_price, exit_time = curr_sl, w_times[i]
                    break
                if c_ask <= tp_price:
                    final_verdict = "TP"
                    exit_price, exit_time = tp_price, w_times[i]
                    break
        
        # R Calculation
        gross_r = ((exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)) / risk_val
        comm_r = COMMISSION_PIPS / risk_pips
        slip_r = SLIPPAGE_PIPS / risk_pips
        
        # Calculate real spread at entry if available, otherwise use default SPREAD_PIPS
        # Since we load the buffer, we can check the spread at w_times[0] (which is close to entry)
        # But for robustness and per instructions, we can use the default 0.30 pips if real not easily accessible
        spread_pips_actual = SPREAD_PIPS
        if len(w_asks) > 0 and len(w_bids) > 0:
            actual = (w_asks[0] - w_bids[0]) * 10000
            if actual > 0.05 and actual < 5.0:  # Sanity check
                spread_pips_actual = actual
        
        spread_r = spread_pips_actual / risk_pips
        net_r = gross_r - (comm_r + slip_r + spread_r)
        
        results.append({
            "trade_id": idx,
            "entry_time": entry_time.isoformat(),
            "exit_time": pd.Timestamp(exit_time).isoformat(),
            "direction": direction,
            "verdict": final_verdict,
            "pnl_gross_r": round(float(gross_r), 4),
            "pnl_net_r": round(float(net_r), 4),
            "commission_r": round(float(comm_r), 4),
            "slippage_r": round(float(slip_r), 4),
            "spread_r": round(float(spread_r), 4),
            "net_r": round(float(net_r), 4) # Keeping this for legacy compatibility in the script
        })

    # Metrics calculation
    df_res = pd.DataFrame([r for r in results if r['verdict'] != "NO_AUDITABLE_NO_TICKS"])
    if len(df_res) == 0:
        return {"month": m_str, "status": "INCONCLUSIVE_NO_DATA"}
        
    auditables = len(df_res)
    total_r_net = df_res['net_r'].sum()
    gross_prof = df_res[df_res['net_r'] > 0]['net_r'].sum()
    gross_loss = abs(df_res[df_res['net_r'] < 0]['net_r'].sum())
    pf = gross_prof / gross_loss if gross_loss > 0 else 99.0
    
    metrics = {
        "month": m_str,
        "sample_total": len(month_trades),
        "auditables": auditables,
        "non_auditables": len(month_trades) - auditables,
        "total_R_net_FTMO": round(total_r_net, 4),
        "expectancy_net_FTMO": round(total_r_net / auditables, 4),
        "PF_net_FTMO": round(pf, 2),
        "verdict_net_FTMO": "NET_FTMO_STRONG" if (pf > 1.5 and (total_r_net/auditables) > 0.1) else ("NET_FTMO_SURVIVES" if total_r_net > 0 else "NET_FTMO_FAILS"),
        "engine_version": ENGINE_VERSION,
        "completed_at": datetime.now().isoformat()
    }
    
    # Save files
    prefix = f"PHASE56O_MONTH_{year}{month:02d}"
    df_res.to_csv(os.path.join(month_dir, f"{prefix}_TRADE_LEVEL.csv"), index=False)
    with open(os.path.join(month_dir, f"{prefix}_METRICS.json"), "w") as f: json.dump(metrics, f, indent=4)
    with open(os.path.join(month_dir, f"{prefix}_DATA_QUALITY.json"), "w") as f: json.dump(dq, f, indent=4)
    
    return metrics

def main():
    print(f"Starting PHASE 56O - Corrected Full Historical Runner")
    
    if not os.path.exists(CHECKPOINT_PATH):
        cp = {"historical_progress": [], "summary": {"engine": ENGINE_VERSION}}
        with open(CHECKPOINT_PATH, "w") as f: json.dump(cp, f, indent=4)
    else:
        with open(CHECKPOINT_PATH, "r") as f: cp = json.load(f)

    # Load trades
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw['entry_time_utc'] = pd.to_datetime(df_raw['entry_time'], utc=True)
    
    # Build month queue
    months_in_data = df_raw['entry_time_utc'].dt.to_period('M').unique().astype(str).tolist()
    months_in_data.sort()
    
    completed_months = [m['month'] for m in cp['historical_progress'] if m.get('status') == 'FORENSIC_COMPLETE']
    
    for m_str in months_in_data:
        if m_str in completed_months:
            print(f"Month {m_str} already completed. Skipping.")
            continue
            
        year, month = map(int, m_str.split('-'))
        print(f"Processing {m_str}...")
        
        result = run_replay_month(year, month, df_raw)
        
        if "total_R_net_FTMO" in result:
            result['status'] = 'FORENSIC_COMPLETE'
            cp['historical_progress'].append(result)
            print(f"  Result: {result['total_R_net_FTMO']} R (Audit: {result['auditables']}/{result['sample_total']})")
        else:
            print(f"  Skipped/Error: {result.get('status')}")
            cp['historical_progress'].append({
                "month": m_str,
                "status": result.get('status', 'ERROR'),
                "completed_at": datetime.now().isoformat()
            })
            
        # Save checkpoint after each month
        with open(CHECKPOINT_PATH, "w") as f: json.dump(cp, f, indent=4)
        
    print("PHASE 56O Execution Finished.")

if __name__ == "__main__":
    main()
