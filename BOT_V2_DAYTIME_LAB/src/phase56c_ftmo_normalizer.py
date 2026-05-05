import os
import pandas as pd
import numpy as np
import json
import glob

# Audit Configuration
COMMISSION_PIPS = 0.5
AUDIT_MONTHS = [
    "2015-01", "2015-02", "2015-03", "2015-04", "2015-05", 
    "2015-06", "2015-07", "2015-10", "2015-11",
    "2017-05", "2017-08", "2020-04", "2024-10", "2025-02", "2025-11"
]

ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB"
TICK_DIR = os.path.join(ROOT, "reports", "manipulante_tick_historical")
RAW_TRADES_PATH = os.path.join(ROOT, "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")

# Load Master Reference for Risk
df_master = pd.read_csv(RAW_TRADES_PATH)
df_master['entry_time_utc'] = pd.to_datetime(df_master['entry_time'], utc=True)
if df_master['risk'].mean() < 0.1:
    df_master['risk'] = df_master['risk'] * 10000
master_risk = df_master[['entry_time_utc', 'risk']].copy()
master_risk = master_risk.rename(columns={'risk': 'original_risk_pips'})
master_risk = master_risk.sort_values('entry_time_utc')

inventory = []
all_trades = []

def process_file(file_path, target_month):
    print(f"Processing {os.path.basename(file_path)} for {target_month}...")
    try:
        df = pd.read_csv(file_path)
        
        # Determine entry_time column
        et_col = None
        for c in ['entry_time_utc', 'entry_time', 'entry_time_ny', 'time', 'entry_date', 'entry_time_original']:
            if c in df.columns: et_col = c; break
            
        if et_col:
            df['entry_time_utc_calc'] = pd.to_datetime(df[et_col], utc=True, errors='coerce')
            df['ym_calc'] = df['entry_time_utc_calc'].dt.strftime('%Y-%m')
        
        # Determine month filter
        df_m = None
        if 'month' in df.columns:
            df['month_str'] = df['month'].astype(str)
            df_m = df[df['month_str'] == target_month].copy()
        elif 'ym_calc' in df.columns:
            df_m = df[df['ym_calc'] == target_month].copy()
        
        if df_m is None or df_m.empty: 
            print(f"  No rows found for {target_month}")
            return None
        
        # Ensure entry_time_utc_calc is present for merge
        if 'entry_time_utc_calc' not in df_m.columns:
             # Try to find it again in the subset if not found globally
             for c in ['entry_time_utc', 'entry_time', 'entry_time_ny', 'time', 'entry_date', 'entry_time_original']:
                if c in df_m.columns:
                    df_m['entry_time_utc_calc'] = pd.to_datetime(df_m[c], utc=True, errors='coerce')
                    break

        # PnL Detection
        r_col = None
        for c in ['tick_R', 'pnl_r', 'final_r', 'R', 'tick_R_base', 'r_result', 'tick_r']:
            if c in df_m.columns: r_col = c; break
        if not r_col: return None
        df_m['tick_R_base'] = df_m[r_col]
        
        # Merge asof
        if 'entry_time_utc_calc' in df_m.columns:
            df_m = df_m.sort_values('entry_time_utc_calc')
            # Rename for merge_asof requirements
            df_m = df_m.rename(columns={'entry_time_utc_calc': 'entry_time_utc_merge'})
            df_m = pd.merge_asof(
                df_m, 
                master_risk.rename(columns={'entry_time_utc': 'entry_time_utc_merge'}), 
                on='entry_time_utc_merge', 
                tolerance=pd.Timedelta('2hour'),
                direction='nearest'
            )
        
        # Risk Final
        df_m['risk_pips_final'] = df_m.get('original_risk_pips', np.nan)
        
        # Fallbacks
        for c in ['risk_pips', 'risk', 'Risk']:
            if c in df_m.columns:
                val = df_m[c].copy()
                # Apply price-to-pips if needed
                if val.mean() < 0.1: val = val * 10000
                df_m['risk_pips_final'] = df_m['risk_pips_final'].fillna(val)
        
        # Calculate from price/sl
        if 'entry_price' in df_m.columns and 'sl' in df_m.columns:
            calc_risk = (df_m['entry_price'] - df_m['sl']).abs() * 10000
            mask = (df_m['risk_pips_final'].isnull()) | (df_m['risk_pips_final'] < 0.1)
            df_m.loc[mask, 'risk_pips_final'] = calc_risk[mask]
        elif 'executable_entry' in df_m.columns and 'sl' in df_m.columns:
            calc_risk = (df_m['executable_entry'] - df_m['sl']).abs() * 10000
            mask = (df_m['risk_pips_final'].isnull()) | (df_m['risk_pips_final'] < 0.1)
            df_m.loc[mask, 'risk_pips_final'] = calc_risk[mask]
        elif 'entry_price_raw' in df_m.columns and 'sl' in df_m.columns:
            calc_risk = (df_m['entry_price_raw'] - df_m['sl']).abs() * 10000
            mask = (df_m['risk_pips_final'].isnull()) | (df_m['risk_pips_final'] < 0.1)
            df_m.loc[mask, 'risk_pips_final'] = calc_risk[mask]

        # Filter
        df_m = df_m[df_m['risk_pips_final'] > 0.1].copy()
        
        if not df_m.empty:
            df_m['commission_R_FTMO'] = COMMISSION_PIPS / df_m['risk_pips_final']
            df_m['tick_R_net_FTMO'] = df_m['tick_R_base'] - df_m['commission_R_FTMO']
            df_m['month'] = target_month
            
            outcome = 'UNKNOWN'
            for c in ['tick_outcome', 'outcome', 'final_outcome', 'status']:
                if c in df_m.columns: outcome = df_m[c]; break
            df_m['tick_outcome'] = outcome
            
            return df_m
    except Exception as e:
        print(f"Error in {os.path.basename(file_path)}: {e}")
    return None

month_map = {
    "2015-01": os.path.join(TICK_DIR, "phase50y_lite", "PHASE50Y_LITE_201501_TRADE_LEVEL.csv"),
    "2015-02": os.path.join(TICK_DIR, "phase56_batches", "batch_201502_201503", "PHASE56_BATCH_201502_TRADE_LEVEL.csv"),
    "2015-03": os.path.join(TICK_DIR, "phase56_batches", "batch_201502_201503", "PHASE56_BATCH_201503_TRADE_LEVEL.csv"),
    "2015-04": os.path.join(TICK_DIR, "phase56_batches", "batch_201504_201505", "PHASE56_BATCH_201504_TRADE_LEVEL.csv"),
    "2015-05": os.path.join(TICK_DIR, "phase56_batches", "batch_201504_201505", "PHASE56_BATCH_201505_TRADE_LEVEL.csv"),
    "2015-06": os.path.join(TICK_DIR, "phase56_batches", "batch_201506_201507", "PHASE56_BATCH_201506_TRADE_LEVEL.csv"),
    "2015-07": os.path.join(TICK_DIR, "phase56_batches", "batch_201506_201507", "PHASE56_BATCH_201507_TRADE_LEVEL.csv"),
    "2015-10": os.path.join(TICK_DIR, "phase50y_lite", "PHASE50Y_LITE_201510_TRADE_LEVEL.csv"),
    "2015-11": os.path.join(TICK_DIR, "phase50y_lite", "PHASE50Y_LITE_201511_TRADE_LEVEL.csv"),
    "2017-05": os.path.join(TICK_DIR, "phase50s_results", "PHASE50S_201705_GEMINI_TRADE_LEVEL.csv"),
    "2017-08": os.path.join(TICK_DIR, "phase50s_results", "PHASE50S_201708_GEMINI_TRADE_LEVEL.csv"),
    "2020-04": os.path.join(TICK_DIR, "phase50s_results", "PHASE50S_202004_GEMINI_TRADE_LEVEL.csv"),
    "2024-10": os.path.join(TICK_DIR, "PHASE50O_MODEL_D_TRADE_LEVEL.csv"),
    "2025-02": os.path.join(TICK_DIR, "phase50y_lite", "PHASE50Y_LITE_202502_TRADE_LEVEL.csv"),
    "2025-11": os.path.join(TICK_DIR, "phase50y_lite", "PHASE50Y_LITE_202511_TRADE_LEVEL.csv")
}

for month in AUDIT_MONTHS:
    f = month_map.get(month)
    exists = f and os.path.exists(f)
    df_res = None
    if exists: df_res = process_file(f, month)
    
    if df_res is not None:
        all_trades.append(df_res)
        inventory.append({"month": month, "trade_level_file": os.path.basename(f), "exists": True, "rows": len(df_res), "usable": "YES"})
    else:
        inventory.append({"month": month, "trade_level_file": os.path.basename(f) if exists else "MISSING", "exists": exists, "rows": 0, "usable": "NO"})

pd.DataFrame(inventory).to_csv(os.path.join(TICK_DIR, "PHASE56C_FTMO_COMMISSION_INPUT_INVENTORY.csv"), index=False)

if all_trades:
    df_net = pd.concat(all_trades)
    df_net[['month', 'tick_R_base', 'risk_pips_final', 'commission_R_FTMO', 'tick_R_net_FTMO', 'tick_outcome']].to_csv(os.path.join(TICK_DIR, "PHASE56C_FTMO_NET_TRADE_LEVEL.csv"), index=False)
    
    monthly_metrics = []
    for month in AUDIT_MONTHS:
        dm = df_net[df_net['month'] == month]
        if dm.empty: continue
        
        sample = len(dm)
        net_r = dm['tick_R_net_FTMO'].sum()
        
        pos_net = dm[dm['tick_R_net_FTMO'] > 0]['tick_R_net_FTMO'].sum()
        neg_net = abs(dm[dm['tick_R_net_FTMO'] < 0]['tick_R_net_FTMO'].sum())
        pf_net = pos_net / neg_net if neg_net > 0 else (pos_net if pos_net > 0 else 0)
        
        monthly_metrics.append({
            "month": month,
            "sample": sample,
            "PF_base": round(float(dm[dm['tick_R_base'] > 0]['tick_R_base'].sum() / abs(dm[dm['tick_R_base'] < 0]['tick_R_base'].sum())) if abs(dm[dm['tick_R_base'] < 0]['tick_R_base'].sum()) > 0 else 0, 2),
            "total_R_base": round(float(dm['tick_R_base'].sum()), 2),
            "PF_net_FTMO": round(float(pf_net), 2),
            "expectancy_net_FTMO": round(float(dm['tick_R_net_FTMO'].mean()), 4),
            "total_R_net_FTMO": round(float(dm['tick_R_net_FTMO'].sum()), 2),
            "avg_commission_R_FTMO": round(float(dm['commission_R_FTMO'].mean()), 4),
            "verdict_net_FTMO": "NET_FTMO_STRONG" if pf_net > 1.5 and dm['tick_R_net_FTMO'].mean() > 0.1 else "NET_FTMO_SURVIVES" if pf_net > 1.2 else "NET_FTMO_FRAGILE" if pf_net > 1.05 else "NET_FTMO_FAILS"
        })
    pd.DataFrame(monthly_metrics).to_csv(os.path.join(TICK_DIR, "PHASE56C_FTMO_NET_MONTHLY_METRICS.csv"), index=False)
    
    pos_net_agg = df_net[df_net['tick_R_net_FTMO'] > 0]['tick_R_net_FTMO'].sum()
    neg_net_agg = abs(df_net[df_net['tick_R_net_FTMO'] < 0]['tick_R_net_FTMO'].sum())
    
    agg = {
        "total_months": len(monthly_metrics),
        "total_trades": len(df_net),
        "PF_base_aggregate": round(float(df_net[df_net['tick_R_base'] > 0]['tick_R_base'].sum() / abs(df_net[df_net['tick_R_base'] < 0]['tick_R_base'].sum())), 2) if abs(df_net[df_net['tick_R_base'] < 0]['tick_R_base'].sum()) > 0 else 0,
        "total_R_base_aggregate": round(float(df_net['tick_R_base'].sum()), 2),
        "PF_net_FTMO_aggregate": round(float(pos_net_agg / neg_net_agg), 2) if neg_net_agg > 0 else 0,
        "total_R_net_FTMO_aggregate": round(float(df_net['tick_R_net_FTMO'].sum()), 2),
        "expectancy_net_FTMO_aggregate": round(float(df_net['tick_R_net_FTMO'].mean()), 4),
        "avg_commission_R_FTMO": round(float(df_net['commission_R_FTMO'].mean()), 4),
        "months_positive_net_FTMO": sum(1 for m in monthly_metrics if m['total_R_net_FTMO'] > 0),
        "months_negative_net_FTMO": sum(1 for m in monthly_metrics if m['total_R_net_FTMO'] <= 0)
    }
    with open(os.path.join(TICK_DIR, "PHASE56C_FTMO_NET_AGGREGATE_METRICS.json"), "w") as f:
        json.dump(agg, f, indent=2)

print("FTMO Normalization completed.")
