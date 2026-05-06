import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import subprocess
import sys

# CONFIGURATION
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
TICK_BASE_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
CHECKPOINT_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\PHASE56_FULL_HISTORICAL_CHECKPOINT.json"
EXTRACTOR_SCRIPT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src\phase50s_resumable_tick_extractor.py"
OUTPUT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches"

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

MONTHS_TO_PROCESS = ["2019-08", "2019-09", "2019-10", "2019-11", "2019-12"]

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def extract_month(year, month):
    print(f"Extracting ticks for {year}-{month:02d}...")
    base_cmd = ["python", EXTRACTOR_SCRIPT, "--year", str(year), "--month", str(month), "--symbol", "EURUSD", "--max-days", "31"]
    if not run_command(base_cmd + ["--mode", "extract"]): return False
    if not run_command(base_cmd + ["--mode", "finalize"]): return False
    if not run_command(base_cmd + ["--mode", "validate"]): return False
    return True

def replay_month(m_str, df_ticks, df_month_trades):
    results = []
    print(f"Replaying {m_str} ({len(df_month_trades)} trades)...")
    
    for i, (_, trade) in enumerate(df_month_trades.iterrows()):
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        sl = trade["sl"]
        tp = trade["tp"]
        side = "LONG" if trade["type"].lower() == "buy" else "SHORT"
        risk_pips = abs(entry_price - sl) * 10000
        commission_r = 0.5 / risk_pips if risk_pips > 0 else 0
        
        end_window = entry_time + timedelta(days=2)
        window = df_ticks[(df_ticks["timestamp_utc"] >= entry_time) & (df_ticks["timestamp_utc"] <= end_window)].copy()
        
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
                    if not be_active and (bid - entry_price) * 10000 >= 0.4 * risk_pips: be_active = True
                    if bid >= tp:
                        outcome, exit_price, exit_time = "TP", tp, t_utc
                        break
                    stop_lvl = entry_price if be_active else sl
                    if bid <= stop_lvl:
                        outcome, exit_price, exit_time = ("BE" if be_active else "SL"), stop_lvl, t_utc
                        break
                else: # SHORT
                    if not be_active and (entry_price - ask) * 10000 >= 0.4 * risk_pips: be_active = True
                    if ask <= tp:
                        outcome, exit_price, exit_time = "TP", tp, t_utc
                        break
                    stop_lvl = entry_price if be_active else sl
                    if ask >= stop_lvl:
                        outcome, exit_price, exit_time = ("BE" if be_active else "SL"), stop_lvl, t_utc
                        break
            
            if exit_price is None:
                outcome = "TIME_EXIT"
                exit_price = bid_values[-1] if side == "LONG" else ask_values[-1]
                exit_time = pd.Timestamp(ts_values[-1], tz='UTC')

            if outcome == "TP": pnl_base = 1.4
            elif outcome == "SL": pnl_base = -1.0
            elif outcome == "BE": pnl_base = 0.0
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
        "results": results,
        "metrics": {
            "month": m_str,
            "sample_total": len(df_res),
            "auditables": len(auditables),
            "non_auditables": len(df_res) - len(auditables),
            "total_R_net_FTMO": round(total_r, 4),
            "expectancy_net_FTMO": round(total_r / len(auditables), 4) if len(auditables) > 0 else 0,
            "PF_net_FTMO": round(auditables[auditables["pnl_net_ftmo"] > 0]["pnl_net_ftmo"].sum() / abs(auditables[auditables["pnl_net_ftmo"] < 0]["pnl_net_ftmo"].sum()), 2) if auditables[auditables["pnl_net_ftmo"] < 0]["pnl_net_ftmo"].sum() != 0 else 99
        }
    }

def main():
    print("[PHASE56M] Starting optimized queue resumption...")
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)
    
    with open(CHECKPOINT_PATH, 'r') as f: checkpoint = json.load(f)

    batch_report = []

    for m in MONTHS_TO_PROCESS:
        year, month = map(int, m.split("-"))
        parquet_path = os.path.join(TICK_BASE_PATH, f"EURUSD_ticks_{year}_{month:02d}.parquet")
        
        if not os.path.exists(parquet_path):
            if not extract_month(year, month):
                print(f"Failed to extract {m}. Skipping.")
                continue
        
        df_ticks = pd.read_parquet(parquet_path)
        df_ticks["timestamp_utc"] = pd.to_datetime(df_ticks["timestamp_utc"], utc=True)
        
        df_month_trades = df_raw[(df_raw["entry_time"].dt.year == year) & (df_raw["entry_time"].dt.month == month)].copy()
        
        res = replay_month(m, df_ticks, df_month_trades)
        metrics = res["metrics"]
        
        # Save Month Outputs
        month_dir = os.path.join(OUTPUT_ROOT, f"batch_201908_201912")
        os.makedirs(month_dir, exist_ok=True)
        
        pd.DataFrame(res["results"]).to_csv(os.path.join(month_dir, f"PHASE56_MONTH_{m.replace('-','')}_TRADE_LEVEL.csv"), index=False)
        with open(os.path.join(month_dir, f"PHASE56_MONTH_{m.replace('-','')}_METRICS.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Update Checkpoint
        checkpoint["historical_progress"] = [x for x in checkpoint["historical_progress"] if x["month"] != m]
        checkpoint["historical_progress"].append({
            **metrics,
            "batch_id": "PHASE56M_RESUMPTION",
            "replay_status": "FORENSIC_COMPLETE",
            "completed_at": datetime.utcnow().isoformat()
        })
        
        batch_report.append(metrics)
        print(f"Completed {m}: {metrics['total_R_net_FTMO']} R")

    # Final Checkpoint Save (marking summary as dirty)
    checkpoint["summary"]["warning"] = "GLOBAL_SUMMARY_REQUIRES_RECONCILIATION"
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=4)

    # Batch Reports
    report_json = {
        "status": "PHASE56M_OPTIMIZED_QUEUE_COMPLETED",
        "timestamp": datetime.utcnow().isoformat(),
        "months_processed": batch_report
    }
    with open(os.path.join(OUTPUT_ROOT, "PHASE56M_201908_201912_OPTIMIZED_QUEUE_REPORT.json"), "w") as f:
        json.dump(report_json, f, indent=4)

    print("[PHASE56M] Batch finished.")

if __name__ == "__main__":
    main()
