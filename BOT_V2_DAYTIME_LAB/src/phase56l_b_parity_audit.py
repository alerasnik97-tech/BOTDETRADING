import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# CONFIGURATION
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
TICK_BASE_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
CHECKPOINT_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\PHASE56_FULL_HISTORICAL_CHECKPOINT.json"
AUDIT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_parity_audit"

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

def run_optimized_replay(month_str, df_ticks_month, df_month_trades):
    results = []
    print(f"Running optimized replay for {month_str}...")
    
    for i, (_, trade) in enumerate(df_month_trades.iterrows()):
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        sl = trade["sl"]
        tp = trade["tp"]
        side = "LONG" if trade["type"].lower() == "buy" else "SHORT"
        risk_pips = abs(entry_price - sl) * 10000
        commission_r = 0.5 / risk_pips if risk_pips > 0 else 0
        
        end_window = entry_time + timedelta(days=2)
        window = df_ticks_month[(df_ticks_month["timestamp_utc"] >= entry_time) & (df_ticks_month["timestamp_utc"] <= end_window)].copy()
        
        outcome = "TIME_EXIT"
        exit_price, exit_time = None, None
        be_active = False
        auditable = "YES"

        if window.empty:
            auditable = "NO"
            pnl_net_ftmo = np.nan
        else:
            ts_values = window["timestamp_utc"].values
            bid_values = window["bid"].values
            ask_values = window["ask"].values
            
            for j in range(len(window)):
                t_utc = pd.Timestamp(ts_values[j], tz='UTC')
                t_ny = t_utc.astimezone(NY)
                bid, ask = bid_values[j], ask_values[j]
                
                if t_ny.hour > 19 or (t_ny.hour == 19 and t_ny.minute >= 45):
                    outcome = "TIME_EXIT"
                    exit_price = bid if side == "LONG" else ask
                    exit_time = t_utc
                    break
                
                if side == "LONG":
                    if not be_active and (bid - entry_price) * 10000 >= 0.4 * risk_pips:
                        be_active = True
                    # Check TP first
                    if bid >= tp:
                        outcome = "TP"
                        exit_price = tp
                        exit_time = t_utc
                        break
                    stop_lvl = entry_price if be_active else sl
                    if bid <= stop_lvl:
                        outcome = "BE" if be_active else "SL"
                        exit_price = stop_lvl
                        exit_time = t_utc
                        break
                else:
                    if not be_active and (entry_price - ask) * 10000 >= 0.4 * risk_pips:
                        be_active = True
                    # Check TP first
                    if ask <= tp:
                        outcome = "TP"
                        exit_price = tp
                        exit_time = t_utc
                        break
                    stop_lvl = entry_price if be_active else sl
                    if ask >= stop_lvl:
                        outcome = "BE" if be_active else "SL"
                        exit_price = stop_lvl
                        exit_time = t_utc
                        break
            
            if exit_price is None:
                outcome = "TIME_EXIT"
                exit_price = bid_values[-1] if side == "LONG" else ask_values[-1]
                exit_time = pd.Timestamp(ts_values[-1], tz='UTC')

            # Calculate PnL matching Phase 56J exactly
            if outcome == "TP":
                pnl_base = 1.4
            elif outcome == "SL":
                pnl_base = -1.0
            elif outcome == "BE":
                pnl_base = 0.0
            else:
                pips = (exit_price - entry_price) * 10000 if side == "LONG" else (entry_price - exit_price) * 10000
                pnl_base = pips / risk_pips if risk_pips > 0 else 0
            
            pnl_net_ftmo = pnl_base - commission_r

        results.append({
            "entry_time": entry_time.isoformat(),
            "side": side,
            "outcome": outcome,
            "pnl_net_ftmo": round(pnl_net_ftmo, 4) if not np.isnan(pnl_net_ftmo) else "NaN",
            "auditable": auditable
        })
    
    df_res = pd.DataFrame(results)
    df_res["pnl_net_ftmo"] = pd.to_numeric(df_res["pnl_net_ftmo"], errors='coerce')
    auditables = df_res[df_res["auditable"] == "YES"]
    total_r = auditables["pnl_net_ftmo"].sum()
    
    return {
        "sample_total": len(df_res),
        "auditables": len(auditables),
        "total_R_net_FTMO": round(total_r, 4),
        "expectancy_net_FTMO": round(total_r / len(auditables), 4) if len(auditables) > 0 else 0,
        "PF_net_FTMO": round(auditables[auditables["pnl_net_ftmo"] > 0]["pnl_net_ftmo"].sum() / abs(auditables[auditables["pnl_net_ftmo"] < 0]["pnl_net_ftmo"].sum()), 2) if auditables[auditables["pnl_net_ftmo"] < 0]["pnl_net_ftmo"].sum() != 0 else 99
    }

def main():
    if not os.path.exists(AUDIT_DIR): os.makedirs(AUDIT_DIR)
    with open(CHECKPOINT_PATH, 'r') as f: checkpoint = json.load(f)
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)

    comparison = {}

    for m in ["2018-09", "2018-04"]:
        year, month = map(int, m.split("-"))
        df_month_trades = df_raw[(df_raw["entry_time"].dt.year == year) & (df_raw["entry_time"].dt.month == month)].copy()
        
        tick_file = os.path.join(TICK_BASE_PATH, f"EURUSD_ticks_{year}_{month:02d}.parquet")
        df_ticks = pd.read_parquet(tick_file)
        df_ticks["timestamp_utc"] = pd.to_datetime(df_ticks["timestamp_utc"], utc=True)
        
        optimized_metrics = run_optimized_replay(m, df_ticks, df_month_trades)
        
        # Get canonical from checkpoint
        canonical = next((x for x in checkpoint["historical_progress"] if x["month"] == m), None)
        
        comparison[m] = {
            "optimized": optimized_metrics,
            "canonical": canonical
        }

    with open(os.path.join(AUDIT_DIR, "parity_results.json"), "w") as f:
        json.dump(comparison, f, indent=4)
    print("Parity audit completed.")

if __name__ == "__main__":
    main()
