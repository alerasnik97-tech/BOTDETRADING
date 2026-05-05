import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import pytz

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

TICK_ROOT = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly"
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
OUTPUT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\batch_201506_201507"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_replay(year, month):
    print(f"--- Replaying {year}-{month:02d} ---")
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)
    df_month = df_raw[(df_raw["entry_time"].dt.year == year) & (df_raw["entry_time"].dt.month == month)].copy()
    
    if df_month.empty:
        print(f"No trades for {year}-{month:02d}")
        return None
        
    tick_file = os.path.join(TICK_ROOT, f"EURUSD_ticks_{year}_{month:02d}.parquet")
    if not os.path.exists(tick_file):
        print(f"Tick file missing: {tick_file}")
        return None
        
    df_ticks = pd.read_parquet(tick_file)
    df_ticks["timestamp_utc"] = pd.to_datetime(df_ticks["timestamp_utc"], utc=True)
    df_ticks.sort_values("timestamp_utc", inplace=True)
    
    results = []
    for _, trade in df_month.iterrows():
        trade_id = trade.get("trade_id", "N/A")
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        sl = trade["sl"]
        tp = trade["tp"]
        side_raw = trade["type"]
        side = "LONG" if side_raw.lower() == "buy" else "SHORT"
        risk_pips = trade["risk"]
        
        # Window: from entry to end of day + 1
        window = df_ticks[(df_ticks["timestamp_utc"] >= entry_time) & (df_ticks["timestamp_utc"] <= entry_time + timedelta(days=2))].copy()
        
        outcome = "TIME_EXIT"
        exit_price = None
        exit_time = None
        be_active = False
        
        for _, tick in window.iterrows():
            t_utc = tick["timestamp_utc"]
            t_ny = t_utc.astimezone(NY)
            bid = tick["bid"]
            ask = tick["ask"]
            
            # Check Time Exit 19:45 NY
            if t_ny.hour > 19 or (t_ny.hour == 19 and t_ny.minute >= 45):
                outcome = "TIME_EXIT"
                exit_price = bid if side == "LONG" else ask
                exit_time = t_utc
                break
                
            if side == "LONG":
                # BE Trigger +0.4R
                if not be_active and (bid - entry_price) * 10000 >= 0.4 * risk_pips:
                    be_active = True
                
                # TP
                if bid >= tp:
                    outcome = "TP"
                    exit_price = tp
                    exit_time = t_utc
                    break
                # SL / BE
                current_sl = entry_price if be_active else sl
                if bid <= current_sl:
                    outcome = "BE" if be_active else "SL"
                    exit_price = current_sl
                    exit_time = t_utc
                    break
            else: # SHORT
                # BE Trigger +0.4R
                if not be_active and (entry_price - ask) * 10000 >= 0.4 * risk_pips:
                    be_active = True
                
                # TP
                if ask <= tp:
                    outcome = "TP"
                    exit_price = tp
                    exit_time = t_utc
                    break
                # SL / BE
                current_sl = entry_price if be_active else sl
                if ask >= current_sl:
                    outcome = "BE" if be_active else "SL"
                    exit_price = current_sl
                    exit_time = t_utc
                    break
        
        if exit_price is None:
            last_tick = window.iloc[-1]
            exit_price = last_tick["bid"] if side == "LONG" else last_tick["ask"]
            exit_time = last_tick["timestamp_utc"]
            outcome = "TIME_EXIT_FALLBACK"

        pnl_r = 0
        if outcome == "TP": pnl_r = 1.4
        elif outcome == "SL": pnl_r = -1.0
        elif outcome == "BE": pnl_r = 0.0
        else:
            if side == "LONG":
                pnl_r = (exit_price - entry_price) * 10000 / risk_pips
            else:
                pnl_r = (entry_price - exit_price) * 10000 / risk_pips
        
        results.append({
            "trade_id": trade_id,
            "entry_time": entry_time.isoformat(),
            "side": side,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "outcome": outcome,
            "exit_price": exit_price,
            "exit_time": exit_time.isoformat(),
            "pnl_r": round(pnl_r, 4)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"PHASE56_BATCH_{year}{month:02d}_TRADE_LEVEL.csv"), index=False)
    
    # Metrics
    pos_r = res_df[res_df["pnl_r"] > 0]["pnl_r"].sum()
    neg_r = abs(res_df[res_df["pnl_r"] < 0]["pnl_r"].sum())
    pf = pos_r / neg_r if neg_r > 0 else (pos_r if pos_r > 0 else 0)
    
    metrics = {
        "year": year,
        "month": month,
        "sample": len(res_df),
        "PF": round(pf, 2),
        "expectancy": round(res_df["pnl_r"].mean(), 4),
        "total_R": round(res_df["pnl_r"].sum(), 2),
        "TP": int((res_df["outcome"] == "TP").sum()),
        "BE": int((res_df["outcome"] == "BE").sum()),
        "SL": int((res_df["outcome"] == "SL").sum()),
        "TIME_EXIT": int(res_df["outcome"].str.contains("TIME_EXIT").sum()),
    }
    
    return metrics

def main():
    m06 = run_replay(2015, 6)
    m07 = run_replay(2015, 7)
    
    all_metrics = []
    if m06: all_metrics.append(m06)
    if m07: all_metrics.append(m07)
    
    with open(os.path.join(OUTPUT_DIR, "PHASE56_BATCH_201506_201507_METRICS.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("Batch replay finished.")

if __name__ == "__main__":
    main()
