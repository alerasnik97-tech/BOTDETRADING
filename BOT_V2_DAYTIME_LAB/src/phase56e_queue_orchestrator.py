import os
import sys
import pandas as pd
import numpy as np
import json
import subprocess
from datetime import datetime, timedelta
import pytz

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Paths
ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB"
TICK_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly"
RAW_TRADES_PATH = os.path.join(ROOT, "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")
CHECKPOINT_PATH = os.path.join(ROOT, "reports", "manipulante_tick_historical", "phase56_batches", "PHASE56_FULL_HISTORICAL_CHECKPOINT.json")
OUTPUT_ROOT = os.path.join(ROOT, "reports", "manipulante_tick_historical", "phase56_batches")
EXTRACTOR_PATH = os.path.join(ROOT, "src", "phase50s_resumable_tick_extractor.py")

QUEUE = [
    {"id": "batch_04", "months": ["2015-08", "2015-09"]},
    {"id": "batch_05", "months": ["2015-12", "2016-01"]},
    {"id": "batch_06", "months": ["2016-02", "2016-03"]},
    {"id": "batch_07", "months": ["2016-04", "2016-05"]},
]

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"historical_progress": [], "summary": {}}

def save_checkpoint(cp):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(cp, f, indent=4)

def ensure_tick_data(year, month):
    tick_file = os.path.join(TICK_ROOT, f"EURUSD_ticks_{year}_{month:02d}.parquet")
    if os.path.exists(tick_file):
        return True, "EXISTS"
    
    print(f"  [EXTRACTING] {year}-{month:02d} started at {datetime.now()}...")
    try:
        # Resume (download missing days)
        subprocess.run([sys.executable, EXTRACTOR_PATH, "--year", str(year), "--month", str(month), "--mode", "resume"])
        # Finalize (concat days to monthly)
        subprocess.run([sys.executable, EXTRACTOR_PATH, "--year", str(year), "--month", str(month), "--mode", "finalize"])
        # Validate
        subprocess.run([sys.executable, EXTRACTOR_PATH, "--year", str(year), "--month", str(month), "--mode", "validate"])
        
        if os.path.exists(tick_file):
            print(f"  [EXTRACTED] {year}-{month:02d} successful.")
            return True, "EXTRACTED"
    except Exception as e:
        print(f"  [ERROR] Extraction failed for {year}-{month}: {e}")
    
    return False, "MISSING"

def run_replay(year, month, batch_dir):
    print(f"  [REPLAY] {year}-{month:02d}...")
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)
    df_month = df_raw[(df_raw["entry_time"].dt.year == year) & (df_raw["entry_time"].dt.month == month)].copy()
    
    if df_month.empty:
        print(f"  [NO_TRADES] {year}-{month:02d}")
        return {"status": "NO_TRADES"}
        
    tick_file = os.path.join(TICK_ROOT, f"EURUSD_ticks_{year}_{month:02d}.parquet")
    df_ticks = pd.read_parquet(tick_file)
    df_ticks["timestamp_utc"] = pd.to_datetime(df_ticks["timestamp_utc"], utc=True)
    df_ticks.sort_values("timestamp_utc", inplace=True)
    
    results = []
    print(f"  Processing {len(df_month)} trades...")
    for _, trade in df_month.iterrows():
        trade_id = trade.get("trade_id", "N/A")
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        sl_initial = trade["sl"]
        tp = trade["tp"]
        side = "LONG" if trade["type"].lower() == "buy" else "SHORT"
        
        # USE THE RISK COLUMN (already includes the initial stop distance)
        risk_pips = trade["risk"] * 10000
        if risk_pips < 1.0: 
            # Fallback to calculation if risk column is empty/0
            risk_pips = abs(entry_price - sl_initial) * 10000
            if risk_pips < 1.0: risk_pips = 10.0 # Extreme fallback
        
        window = df_ticks[(df_ticks["timestamp_utc"] >= entry_time) & (df_ticks["timestamp_utc"] <= entry_time + timedelta(days=2))].copy()
        
        outcome = "TIME_EXIT"
        exit_price = None
        exit_time = None
        be_active = False
        
        for _, tick in window.iterrows():
            t_utc = tick["timestamp_utc"]
            t_ny = t_utc.astimezone(NY)
            bid = tick["bid"]; ask = tick["ask"]
            
            if t_ny.hour > 19 or (t_ny.hour == 19 and t_ny.minute >= 45):
                outcome = "TIME_EXIT"; exit_price = bid if side == "LONG" else ask; exit_time = t_utc; break
                
            if side == "LONG":
                if not be_active and (bid - entry_price) * 10000 >= 0.4 * risk_pips: be_active = True
                if bid >= tp: outcome = "TP"; exit_price = tp; exit_time = t_utc; break
                current_sl = entry_price if be_active else sl_initial
                if bid <= current_sl: outcome = "BE" if be_active else "SL"; exit_price = current_sl; exit_time = t_utc; break
            else:
                if not be_active and (entry_price - ask) * 10000 >= 0.4 * risk_pips: be_active = True
                if ask <= tp: outcome = "TP"; exit_price = tp; exit_time = t_utc; break
                current_sl = entry_price if be_active else sl_initial
                if ask >= current_sl: outcome = "BE" if be_active else "SL"; exit_price = current_sl; exit_time = t_utc; break
        
        if exit_price is None and not window.empty:
            last = window.iloc[-1]; exit_price = last["bid"] if side == "LONG" else last["ask"]; exit_time = last["timestamp_utc"]; outcome = "TIME_EXIT_FALLBACK"

        pnl_r = 0
        if outcome == "TP": pnl_r = 1.4
        elif outcome == "SL": pnl_r = -1.0
        elif outcome == "BE": pnl_r = 0.0
        elif exit_price:
            pnl_r = ((exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)) * 10000 / risk_pips
        
        # FTMO Commission: 0.5 pips round-turn
        comm_r = 0.5 / risk_pips
        pnl_r_net = pnl_r - comm_r
        
        results.append({
            "trade_id": trade_id, "entry_time": entry_time.isoformat(), "side": side,
            "outcome": outcome, "pnl_r_base": round(pnl_r, 4), "commission_r": round(comm_r, 4), "pnl_r_net": round(pnl_r_net, 4)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(batch_dir, f"PHASE56_BATCH_{year}{month:02d}_TRADE_LEVEL.csv"), index=False)
    
    pos_r = res_df[res_df["pnl_r_base"] > 0]["pnl_r_base"].sum()
    neg_r = abs(res_df[res_df["pnl_r_base"] < 0]["pnl_r_base"].sum())
    pf_base = pos_r / neg_r if neg_r > 0 else (pos_r if pos_r > 0 else 0)
    
    pos_r_net = res_df[res_df["pnl_r_net"] > 0]["pnl_r_net"].sum()
    neg_r_net = abs(res_df[res_df["pnl_r_net"] < 0]["pnl_r_net"].sum())
    pf_net = pos_r_net / neg_r_net if neg_r_net > 0 else (pos_r_net if pos_r_net > 0 else 0)
    
    exp_net = res_df["pnl_r_net"].mean()
    verdict_net = "NET_FTMO_STRONG" if pf_net > 1.5 and exp_net > 0.1 else "NET_FTMO_SURVIVES" if pf_net > 1.2 else "NET_FTMO_FRAGILE" if pf_net > 1.05 else "NET_FTMO_FAILS"

    metrics = {
        "month": f"{year}-{month:02d}", "sample": len(res_df), "PF_base": round(pf_base, 2), 
        "expectancy_base": round(res_df["pnl_r_base"].mean(), 4), "total_R_base": round(res_df["pnl_r_base"].sum(), 2),
        "PF_net_FTMO": round(pf_net, 2), "expectancy_net_FTMO": round(exp_net, 4), "total_R_net_FTMO": round(res_df["pnl_r_net"].sum(), 2),
        "avg_commission_R": round(res_df["commission_r"].mean(), 4), "verdict_net_FTMO": verdict_net
    }
    print(f"    [RESULT] {metrics['month']}: PF_net={metrics['PF_net_FTMO']}, Total_R_net={metrics['total_R_net_FTMO']}")
    return metrics

def process_queue():
    cp = load_checkpoint()
    final_report = {"batches_completed": [], "batches_blocked": [], "months_processed": []}
    
    for batch in QUEUE:
        batch_id = batch["id"]
        # Allow re-processing if commission is obviously wrong (heuristic: avg_commission_R > 0.3)
        # 0.3R is too high for a single trade average commission (usually 0.02 - 0.05)
        already_done = batch_id in cp and cp[batch_id].get("status") == "COMPLETED"
        if already_done:
            # Check if any month in historical_progress for this batch has bad commission
            months = batch["months"]
            bad_data = False
            for p in cp["historical_progress"]:
                if p["month"] in months and p.get("avg_commission_R", 0) > 0.1:
                    bad_data = True; break
            
            if not bad_data:
                print(f"[SKIP] Batch {batch_id} already completed correctly.")
                continue
            else:
                print(f"[RE-RUN] Batch {batch_id} has corrupted commission data. Correcting...")
                # Remove bad entries
                cp["historical_progress"] = [p for p in cp["historical_progress"] if not (p["month"] in months and p.get("avg_commission_R", 0) > 0.1)]
                del cp[batch_id]
                save_checkpoint(cp)

        months = batch["months"]
        batch_dir_name = f"batch_{months[0].replace('-','')}_{months[1].replace('-','')}"
        batch_dir = os.path.join(OUTPUT_ROOT, batch_dir_name)
        os.makedirs(batch_dir, exist_ok=True)
        
        batch_results = []
        print(f"--- Processing {batch_id}: {months} at {datetime.now()} ---")
        
        for m_str in months:
            # If month still exists in progress, skip (might happen if re-run only part of batch)
            if any(p["month"] == m_str for p in cp["historical_progress"]):
                print(f"  [SKIP] Month {m_str} already audited.")
                continue
            
            y, m = map(int, m_str.split("-"))
            ok, msg = ensure_tick_data(y, m)
            if not ok:
                print(f"  [BLOCKED] No tick data for {m_str}")
                continue
                
            metrics = run_replay(y, m, batch_dir)
            if metrics.get("status") == "NO_TRADES":
                metrics = {"month": m_str, "sample": 0, "PF_base": 0, "expectancy_base": 0, "total_R_base": 0, "verdict_net_FTMO": "NO_TRADES"}
            
            metrics["batch_id"] = batch_id
            metrics["completed_at"] = datetime.now().isoformat()
            cp["historical_progress"].append(metrics)
            batch_results.append(metrics)
            final_report["months_processed"].append(metrics)
            
        cp[batch_id] = {"months": months, "status": "COMPLETED", "timestamp": datetime.now().isoformat()}
        save_checkpoint(cp)
        final_report["batches_completed"].append(batch_id)
        
        # Save batch specific metrics
        with open(os.path.join(batch_dir, f"PHASE56_{batch_id.upper()}_FTMO_NET.json"), "w") as f:
            json.dump(batch_results, f, indent=4)

    # Summary Update
    all_p = cp["historical_progress"]
    cp["summary"] = {
        "total_months": len(all_p),
        "total_sample": sum(p["sample"] for p in all_p),
        "total_R_base": round(sum(p.get("total_R_base", 0) for p in all_p), 2),
        "total_R_net_FTMO": round(sum(p.get("total_R_net_FTMO", 0) for p in all_p), 2),
        "expectancy_net_FTMO": round(np.mean([p.get("expectancy_net_FTMO", 0) for p in all_p if p.get("sample", 0) > 0]), 4)
    }
    save_checkpoint(cp)
    
    with open(os.path.join(OUTPUT_ROOT, "PHASE56E_OVERNIGHT_QUEUE_REPORT.json"), "w") as f:
        json.dump(final_report, f, indent=4)
    
    print(f"\nQueue execution finished at {datetime.now()}.")
    return final_report

if __name__ == "__main__":
    process_queue()
