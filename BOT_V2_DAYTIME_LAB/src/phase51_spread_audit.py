import pandas as pd
import numpy as np
import os
import json
import pytz

# Paths
ROOT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
REPORTS_DIR = os.path.join(ROOT_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")
TICK_DIR = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly"
RAW_TRADES_PATH = os.path.join(ROOT_DIR, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")

TARGET_MONTHS = ["2015-01", "2015-10", "2015-11", "2017-05", "2017-08", "2020-04", "2024-10", "2025-02", "2025-11"]
UTC = pytz.UTC

def audit_costs():
    print("--- Loading Raw Trades ---")
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw['ym'] = pd.to_datetime(df_raw['entry_time'], utc=True).dt.strftime('%Y-%m')
    df_group_b = df_raw[df_raw['ym'].isin(TARGET_MONTHS)].copy()
    
    print(f"Sample: {len(df_group_b)}")
    
    results = []
    
    for ym in TARGET_MONTHS:
        month_trades = df_group_b[df_group_b['ym'] == ym]
        if month_trades.empty: continue
        
        parquet_file = os.path.join(TICK_DIR, f"EURUSD_ticks_{ym.replace('-', '_')}.parquet")
        if not os.path.exists(parquet_file):
            print(f"WARNING: Missing tick data for {ym}")
            continue
            
        print(f"Processing {ym}...")
        df_ticks = pd.read_parquet(parquet_file)
        if df_ticks['timestamp_utc'].dt.tz is not None:
            df_ticks['timestamp_utc'] = df_ticks['timestamp_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
        df_ticks.set_index('timestamp_utc', inplace=True)
        
        for idx, trade in month_trades.iterrows():
            entry_time = pd.to_datetime(trade['entry_time']).replace(tzinfo=None)
            
            # Initial Risk Pips (from 'risk' column)
            risk_val = trade['risk']
            risk_pips = risk_val * 10000
            
            if risk_pips <= 0:
                print(f"WARNING: Zero risk for trade {idx} in {ym}")
                continue

            # Observed Spread at Entry
            try:
                # Find closest tick at or after entry
                entry_tick = df_ticks.asof(entry_time) # Last tick before or at
                if entry_tick is None:
                    # Try first tick after
                    entry_tick = df_ticks.loc[entry_time:].head(1).iloc[0]
                
                spread_val = entry_tick['ask'] - entry_tick['bid']
                spread_pips = spread_val * 10000
                spread_r = spread_pips / risk_pips
            except:
                spread_pips = 0.4 # Default conservative
                spread_r = 0.4 / risk_pips
            
            results.append({
                "month": ym,
                "trade_idx": idx,
                "direction": trade['type'],
                "risk_pips": round(risk_pips, 4),
                "spread_pips": round(spread_pips, 4),
                "spread_r": round(spread_r, 4),
                "r_result_raw": trade['r_result']
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(REPORTS_DIR, "PHASE51_GROUP_B_SPREAD_COSTS.csv"), index=False)
    
    # Stats
    stats = {
        "risk_pips": {
            "mean": df_res['risk_pips'].mean(),
            "median": df_res['risk_pips'].median(),
            "p10": df_res['risk_pips'].quantile(0.1),
            "p90": df_res['risk_pips'].quantile(0.9)
        },
        "spread_pips": {
            "mean": df_res['spread_pips'].mean(),
            "median": df_res['spread_pips'].median(),
            "p95": df_res['spread_pips'].quantile(0.95)
        },
        "spread_r": {
            "mean": df_res['spread_r'].mean(),
            "median": df_res['spread_r'].median(),
            "p95": df_res['spread_r'].quantile(0.95)
        }
    }
    
    with open(os.path.join(REPORTS_DIR, "PHASE51_GROUP_B_COST_STATS.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Cost stats generated.")

if __name__ == "__main__":
    audit_costs()
