import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import pytz

# CONFIGURATION
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
DATA_DIR = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly"
REPORT_DIR = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical")
REGRESSION_DIR = os.path.join(REPORT_DIR, "phase56_engine_regression")
RAW_TRADES_PATH = os.path.join(REPORT_DIR, "PHASE56_RAW_TRADES_CANONICAL.csv")
CHECKPOINT_PATH = os.path.join(REPORT_DIR, r"phase56_batches\PHASE56_FULL_HISTORICAL_CHECKPOINT.json")

# CONSTANTS
TP_R = 1.4
BE_TRIGGER_R = 0.4
COMMISSION_PIPS = 0.5 

TARGET_MONTHS = [
    (2018, 9),  # Strong month
    (2018, 4),  # Negative month
    (2019, 7),  # Recovered month
    (2020, 1),  # Winter (DST gap check)
    (2020, 2),  # Winter (DST gap check)
    (2017, 8),  # Inconclusive (non-auditables)
    (2016, 8),  # Strong ancient
    (2016, 7)   # Negative ancient
]

def run_corrected_replay(year, month):
    m_str = f"{year}-{month:02d}"
    parquet_path = os.path.join(DATA_DIR, f"EURUSD_ticks_{year}_{month:02d}.parquet")
    
    if not os.path.exists(parquet_path):
        return {"error": f"Parquet not found: {parquet_path}"}
        
    df_ticks = pd.read_parquet(parquet_path)
    ts_col = 'timestamp_utc' if 'timestamp_utc' in df_ticks.columns else 'timestamp'
    df_ticks['timestamp'] = pd.to_datetime(df_ticks[ts_col], utc=True)
    df_ticks = df_ticks.sort_values('timestamp')
    
    tick_times = df_ticks['timestamp'].values
    tick_bids = df_ticks['bid'].values
    tick_asks = df_ticks['ask'].values
    
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw['entry_time'] = pd.to_datetime(df_raw['entry_time'], utc=True)
    
    month_trades = df_raw[
        (df_raw['entry_time'].dt.year == year) & 
        (df_raw['entry_time'].dt.month == month)
    ].copy()
    
    results = []
    ny_tz = pytz.timezone('America/New_York')
    
    for _, trade in month_trades.iterrows():
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        direction = trade['type'] if 'type' in trade else trade['direction']
        risk_val = trade['risk']
        
        # Initial SL reconstruction (Fix from 56N-B)
        sl = entry_price - risk_val if direction == 'LONG' else entry_price + risk_val
        risk_pips = abs(entry_price - sl) * 10000
        
        if risk_pips < 0.1:
            continue
            
        # Proper 19:45 NY time exit (Fix from 56N-B)
        entry_ny = entry_time.tz_convert(ny_tz)
        exit_ny = entry_ny.replace(hour=19, minute=45, second=0, microsecond=0)
        time_exit_utc = exit_ny.tz_convert('UTC')
        
        start_idx = np.searchsorted(tick_times, entry_time.to_datetime64())
        end_limit_idx = np.searchsorted(tick_times, time_exit_utc.to_datetime64())
        
        window_times = tick_times[start_idx:end_limit_idx]
        window_bids = tick_bids[start_idx:end_limit_idx]
        window_asks = tick_asks[start_idx:end_limit_idx]
        
        if len(window_times) == 0:
            results.append({"verdict": "NO_AUDITABLE_NO_TICKS"})
            continue
            
        tp_price = entry_price + (TP_R * (entry_price - sl)) if direction == 'LONG' else entry_price - (TP_R * (sl - entry_price))
        be_trigger_price = entry_price + (BE_TRIGGER_R * (entry_price - sl)) if direction == 'LONG' else entry_price - (BE_TRIGGER_R * (sl - entry_price))
        
        final_verdict = "TIME_EXIT"
        exit_price = window_bids[-1] if direction == 'LONG' else window_asks[-1]
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
                    break
                if curr_bid >= tp_price:
                    final_verdict = "TP"
                    exit_price = tp_price
                    break
            else:
                if not be_active and curr_ask <= be_trigger_price: be_active = True
                current_sl = entry_price if be_active else sl
                if curr_ask >= current_sl:
                    final_verdict = "BE_STOP" if be_active else "SL"
                    exit_price = current_sl
                    break
                if curr_ask <= tp_price:
                    final_verdict = "TP"
                    exit_price = tp_price
                    break
        
        gross_r = (exit_price - entry_price) / abs(entry_price - sl) if direction == 'LONG' else (entry_price - exit_price) / abs(entry_price - sl)
        comm_r = COMMISSION_PIPS / risk_pips
        net_r = gross_r - comm_r
        
        results.append({
            "entry_time": entry_time.isoformat(),
            "verdict": final_verdict,
            "net_r": round(float(net_r), 4),
            "legacy_r": trade.get('r_result', 0)
        })

    df_res = pd.DataFrame([r for r in results if r['verdict'] != "NO_AUDITABLE_NO_TICKS"])
    
    # Trace differences for debugging
    if (year == 2018 and month == 9) or (year == 2018 and month == 4):
        print(f"\nDEBUG: {year}-{month:02d} Trade Comparison")
        for _, r in df_res.iterrows():
            print(f"Time: {r['entry_time']} | New: {r['net_r']} | Legacy: {r['legacy_r']} | Verdict: {r['verdict']}")
    
    auditables = len(df_res)
    total_trades = len(month_trades)
    
    if auditables > 0:
        total_r_net = df_res['net_r'].sum()
        losses = len(df_res[df_res['net_r'] < 0])
        pf = abs(df_res[df_res['net_r'] > 0]['net_r'].sum() / df_res[df_res['net_r'] < 0]['net_r'].sum()) if losses > 0 else 99.0
        
        return {
            "month": m_str,
            "sample_total": total_trades,
            "auditables": auditables,
            "non_auditables": total_trades - auditables,
            "total_R_net_FTMO": round(total_r_net, 4),
            "expectancy_net_FTMO": round(total_r_net / auditables, 4),
            "PF_net_FTMO": round(pf, 2)
        }
    else:
        return {
            "month": m_str,
            "sample_total": total_trades,
            "auditables": 0,
            "non_auditables": total_trades
        }

def run_regression():
    if not os.path.exists(REGRESSION_DIR):
        os.makedirs(REGRESSION_DIR)
        
    with open(CHECKPOINT_PATH, "r") as f:
        checkpoint = json.load(f)
    
    historical = checkpoint.get("historical_progress", [])
    
    comparison_results = []
    
    for year, month in TARGET_MONTHS:
        m_str = f"{year}-{month:02d}"
        print(f"Auditing {m_str}...")
        
        # Current result from checkpoint
        current_cp = next((m for m in historical if m['month'] == m_str), None)
        
        # Run corrected replay
        new_res = run_corrected_replay(year, month)
        
        if "error" in new_res:
            print(f"Skipping {m_str}: {new_res['error']}")
            continue
            
        diffs = {}
        classification = "UNKNOWN"
        
        if current_cp:
            # Map canonical names if they differ
            cp_auditables = current_cp.get('auditables', current_cp.get('sample', 0))
            cp_total_r = current_cp.get('total_R_net_FTMO', current_cp.get('total_R', 0))
            cp_expectancy = current_cp.get('expectancy_net_FTMO', current_cp.get('expectancy', 0))
            cp_pf = current_cp.get('PF_net_FTMO', current_cp.get('PF', 0))
            
            diff_r = abs(new_res['total_R_net_FTMO'] - cp_total_r) if 'total_R_net_FTMO' in new_res else 0
            diff_exp = abs(new_res['expectancy_net_FTMO'] - cp_expectancy) if 'expectancy_net_FTMO' in new_res else 0
            diff_pf = abs(new_res['PF_net_FTMO'] - cp_pf) if 'PF_net_FTMO' in new_res else 0
            
            # Impact Classification
            if new_res['auditables'] != cp_auditables:
                classification = "RESTATED_BY_VALID_BUGFIX" # Coverage improved/changed
            elif diff_r <= 0.02 and diff_exp <= 0.002:
                classification = "UNCHANGED"
            elif diff_r <= 0.1:
                classification = "MINOR_NUMERIC_DIFF"
            else:
                classification = "REQUIRES_MANUAL_REVIEW"
                
            diffs = {
                "total_R_diff": round(new_res.get('total_R_net_FTMO', 0) - cp_total_r, 4),
                "expectancy_diff": round(new_res.get('expectancy_net_FTMO', 0) - cp_expectancy, 4),
                "auditables_diff": new_res['auditables'] - cp_auditables
            }
        
        comparison_results.append({
            "month": m_str,
            "checkpoint": {
                "auditables": cp_auditables if current_cp else None,
                "total_R": cp_total_r if current_cp else None,
                "verdict": current_cp.get('verdict_net_FTMO', current_cp.get('verdict', 'N/A')) if current_cp else None
            },
            "corrected_engine": {
                "auditables": new_res['auditables'],
                "total_R": new_res.get('total_R_net_FTMO', 0),
                "PF": new_res.get('PF_net_FTMO', 0)
            },
            "diffs": diffs,
            "classification": classification
        })

    # Save report
    report_path = os.path.join(REGRESSION_DIR, f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"\nRegression Report saved to: {report_path}")
    
    # Summary to console
    print("\n" + "="*50)
    print(f"{'MONTH':<10} | {'AUDIT_DIFF':<10} | {'R_DIFF':<10} | {'STATUS'}")
    print("-" * 50)
    for res in comparison_results:
        m = res['month']
        a_diff = res['diffs'].get('auditables_diff', 'N/A')
        r_diff = res['diffs'].get('total_R_diff', 'N/A')
        status = res['classification']
        print(f"{m:<10} | {str(a_diff):<10} | {str(r_diff):<10} | {status}")
    print("="*50)

if __name__ == "__main__":
    run_regression()
