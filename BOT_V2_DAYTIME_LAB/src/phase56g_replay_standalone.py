import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime, timedelta
import pytz

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

TICK_ROOT = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly"
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"

def run_replay(year, month, output_dir):
    print(f"--- Replaying {year}-{month:02d} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Trades
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)
    df_month = df_raw[(df_raw["entry_time"].dt.year == year) & (df_raw["entry_time"].dt.month == month)].copy()
    
    if df_month.empty:
        print(f"No trades for {year}-{month:02d}")
        return None
        
    # 2. Load Ticks
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
        risk_pips = trade["risk"] * 10000 # Convert price units to pips
        
        # Window: entry to +2 days (to ensure we cover the 19:45 NY exit)
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
            if not window.empty:
                last_tick = window.iloc[-1]
                exit_price = last_tick["bid"] if side == "LONG" else last_tick["ask"]
                exit_time = last_tick["timestamp_utc"]
                outcome = "TIME_EXIT_FALLBACK"
            else:
                outcome = "ERROR_NO_DATA"
                exit_price = entry_price
                exit_time = entry_time

        # PnL Calculation
        pnl_r = 0
        if outcome == "TP": pnl_r = 1.4
        elif outcome == "SL": pnl_r = -1.0
        elif outcome == "BE": pnl_r = 0.0
        else:
            if side == "LONG":
                pnl_r = (exit_price - entry_price) * 10000 / risk_pips
            else:
                pnl_r = (entry_price - exit_price) * 10000 / risk_pips
        
        # FTMO Commission: $5/lot round-turn = 0.5 pips.
        # commission_R = 0.5 / risk_pips
        comm_r = 0.5 / risk_pips
        pnl_r_net = pnl_r - comm_r
        
        results.append({
            "trade_id": trade_id,
            "entry_time": entry_time.isoformat(),
            "side": side,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "risk_pips": risk_pips,
            "outcome": outcome,
            "exit_price": exit_price,
            "exit_time": exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time),
            "pnl_r_base": round(pnl_r, 4),
            "pnl_r_net_ftmo": round(pnl_r_net, 4),
            "commission_r": round(comm_r, 4)
        })
        
    res_df = pd.DataFrame(results)
    month_id = f"{year}{month:02d}"
    res_df.to_csv(os.path.join(output_dir, f"PHASE56_BATCH_{month_id}_TRADE_LEVEL.csv"), index=False)
    
    # Metrics
    pos_r = res_df[res_df["pnl_r_base"] > 0]["pnl_r_base"].sum()
    neg_r = abs(res_df[res_df["pnl_r_base"] < 0]["pnl_r_base"].sum())
    pf_base = pos_r / neg_r if neg_r > 0 else (pos_r if pos_r > 0 else 0)
    
    pos_r_net = res_df[res_df["pnl_r_net_ftmo"] > 0]["pnl_r_net_ftmo"].sum()
    neg_r_net = abs(res_df[res_df["pnl_r_net_ftmo"] < 0]["pnl_r_net_ftmo"].sum())
    pf_net = pos_r_net / neg_r_net if neg_r_net > 0 else (pos_r_net if pos_r_net > 0 else 0)
    
    # Verdict
    verdict = "NET_FTMO_FAILS"
    if pf_net >= 2.0: verdict = "NET_FTMO_STRONG"
    elif pf_net >= 1.5: verdict = "NET_FTMO_SURVIVES"
    elif pf_net >= 1.0: verdict = "NET_FTMO_FRAGILE"
    
    metrics = {
        "month": f"{year}-{month:02d}",
        "sample": len(res_df),
        "PF_base": round(pf_base, 2),
        "expectancy_base": round(res_df["pnl_r_base"].mean(), 4),
        "total_R_base": round(res_df["pnl_r_base"].sum(), 2),
        "PF_net_FTMO": round(pf_net, 2),
        "expectancy_net_FTMO": round(res_df["pnl_r_net_ftmo"].mean(), 4),
        "total_R_net_FTMO": round(res_df["pnl_r_net_ftmo"].sum(), 2),
        "avg_commission_R": round(res_df["commission_r"].mean(), 4),
        "TP": int((res_df["outcome"] == "TP").sum()),
        "BE": int((res_df["outcome"] == "BE").sum()),
        "SL": int((res_df["outcome"] == "SL").sum()),
        "TIME_EXIT": int(res_df["outcome"].str.contains("TIME_EXIT").sum()),
        "verdict_net_FTMO": verdict
    }
    
    with open(os.path.join(output_dir, f"PHASE56_MONTH_{month_id}_FTMO_NET.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    run_replay(args.year, args.month, args.output_dir)
