import os
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import argparse

# CONFIGURATION
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
TICK_ROOT = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly"
CHECKPOINT_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\PHASE56_FULL_HISTORICAL_CHECKPOINT.json"
OUTPUT_BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches"
LIVE_STATUS_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\PHASE56_LIVE_STATUS.txt"

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

# QUEUE DEFINITION
QUEUE = [
    {"id": "BATCH_13", "months": ["2017-03", "2017-04"]},
    {"id": "BATCH_14", "months": ["2017-05", "2017-06"]},
    {"id": "BATCH_15", "months": ["2017-07", "2017-08"]},
    {"id": "BATCH_16", "months": ["2017-09", "2017-10"]},
    {"id": "BATCH_17", "months": ["2017-11", "2017-12"]},
    {"id": "BATCH_18", "months": ["2018-01", "2018-02"]},
    {"id": "BATCH_19", "months": ["2018-03", "2018-04"]},
    {"id": "BATCH_20", "months": ["2018-05", "2018-06"]},
    {"id": "BATCH_21", "months": ["2018-07", "2018-08"]},
    {"id": "BATCH_22", "months": ["2018-09", "2018-10"]},
    {"id": "BATCH_23", "months": ["2018-11", "2018-12"]},
    {"id": "BATCH_24", "months": ["2019-01", "2019-02"]},
    {"id": "BATCH_25", "months": ["2019-03", "2019-04"]},
    {"id": "BATCH_26", "months": ["2019-05", "2019-06"]},
    {"id": "BATCH_27", "months": ["2019-07", "2019-08"]},
    {"id": "BATCH_28", "months": ["2019-09", "2019-10"]},
    {"id": "BATCH_29", "months": ["2019-11", "2019-12"]},
    {"id": "BATCH_30", "months": ["2020-01", "2020-02"]},
    {"id": "BATCH_31", "months": ["2020-03", "2020-04"]},
    {"id": "BATCH_32", "months": ["2020-05", "2020-06"]},
    {"id": "BATCH_33", "months": ["2020-07", "2020-08"]},
    {"id": "BATCH_34", "months": ["2020-09", "2020-10"]},
    {"id": "BATCH_35", "months": ["2020-11", "2020-12"]},
    {"id": "BATCH_36", "months": ["2021-01", "2021-02"]},
    {"id": "BATCH_37", "months": ["2021-03", "2021-04"]},
]

def update_live_status(batch_id, month, phase, last_completed="NONE", next_step="STARTING"):
    with open(LIVE_STATUS_PATH, "w") as f:
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"BATCH_ACTUAL: {batch_id}\n")
        f.write(f"MES_ACTUAL: {month}\n")
        f.write(f"FASE_ACTUAL: {phase}\n")
        f.write(f"ULTIMO_MES: {last_completed}\n")
        f.write(f"PROXIMO_PASO: {next_step}\n")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return {"historical_progress": [], "summary": {}}

def save_checkpoint(cp):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(cp, f, indent=4)

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result

def process_month(batch_id, m_str, batch_dir):
    update_live_status(batch_id, m_str, "PREFLIGHT")
    year, month = map(int, m_str.split('-'))
    m_clean = m_str.replace('-', '')
    
    # 1. Parquet Rule
    parquet_path = os.path.join(TICK_ROOT, f"EURUSD_ticks_{year}_{month:02d}.parquet")
    if not os.path.exists(parquet_path):
        update_live_status(batch_id, m_str, "EXTRACTION")
        print(f"[{m_str}] Extracting...")
        run_command(f"python src/phase50s_resumable_tick_extractor.py --year {year} --month {month} --mode resume")
        run_command(f"python src/phase50s_resumable_tick_extractor.py --year {year} --month {month} --mode finalize")
        run_command(f"python src/phase50s_resumable_tick_extractor.py --year {year} --month {month} --mode validate")
        if not os.path.exists(parquet_path):
            print(f"[ERROR] Extraction failed for {m_str}")
            return False
    
    # 2. Replay
    update_live_status(batch_id, m_str, "REPLAY")
    print(f"[{m_str}] Replaying...")
    
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)
    df_month = df_raw[(df_raw["entry_time"].dt.year == year) & (df_raw["entry_time"].dt.month == month)].copy()
    
    if df_month.empty:
        # Mark in checkpoint as NO_TRADES
        cp = load_checkpoint()
        cp["historical_progress"].append({
            "month": m_str, "batch_id": batch_id, "status": "NO_TRADES", "completed_at": datetime.utcnow().isoformat()
        })
        save_checkpoint(cp)
        return True

    df_ticks = pd.read_parquet(parquet_path)
    df_ticks["timestamp_utc"] = pd.to_datetime(df_ticks["timestamp_utc"], utc=True)
    df_ticks.sort_values("timestamp_utc", inplace=True)

    results = []
    for _, trade in df_month.iterrows():
        trade_id = trade.get("trade_id", "N/A")
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        sl = trade["sl"]
        tp = trade["tp"]
        side = "LONG" if trade["type"].lower() == "buy" else "SHORT"
        risk_pips = abs(entry_price - sl) * 10000
        
        commission_r = 0.5 / risk_pips if risk_pips > 0 else 0
        
        window = df_ticks[(df_ticks["timestamp_utc"] >= entry_time) & (df_ticks["timestamp_utc"] <= entry_time + timedelta(days=2))].copy()
        
        outcome = "TIME_EXIT"
        exit_price, exit_time = None, None
        be_active = False
        
        for _, tick in window.iterrows():
            t_utc = tick["timestamp_utc"]
            t_ny = t_utc.astimezone(NY)
            bid, ask = tick["bid"], tick["ask"]
            
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
                current_sl = entry_price if be_active else sl
                if bid <= current_sl:
                    outcome, exit_price, exit_time = ("BE" if be_active else "SL"), current_sl, t_utc
                    break
            else:
                if not be_active and (entry_price - ask) * 10000 >= 0.4 * risk_pips: be_active = True
                if ask <= tp:
                    outcome, exit_price, exit_time = "TP", tp, t_utc
                    break
                current_sl = entry_price if be_active else sl
                if ask >= current_sl:
                    outcome, exit_price, exit_time = ("BE" if be_active else "SL"), current_sl, t_utc
                    break
        
        if exit_price is None:
            last_tick = window.iloc[-1]
            exit_price = last_tick["bid"] if side == "LONG" else last_tick["ask"]
            exit_time = last_tick["timestamp_utc"]
            outcome = "TIME_EXIT_FALLBACK"

        pnl_base = 1.4 if outcome == "TP" else (-1.0 if outcome == "SL" else (0.0 if outcome == "BE" else 
                  ((exit_price - entry_price) * 10000 / risk_pips if side == "LONG" else (entry_price - exit_price) * 10000 / risk_pips)))
        
        results.append({
            "trade_id": trade_id, "entry_time": entry_time.isoformat(), "side": side,
            "outcome": outcome, "pnl_base": round(pnl_base, 4), "pnl_net_ftmo": round(pnl_base - commission_r, 4),
            "commission_r": round(commission_r, 4)
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(batch_dir, f"PHASE56_MONTH_{m_clean}_TRADE_LEVEL.csv"), index=False)
    
    # 3. Metrics
    update_live_status(batch_id, m_str, "METRICS")
    total_r_base = res_df["pnl_base"].sum()
    total_r_net = res_df["pnl_net_ftmo"].sum()
    
    pos_base = res_df[res_df["pnl_base"] > 0]["pnl_base"].sum()
    neg_base = abs(res_df[res_df["pnl_base"] < 0]["pnl_base"].sum())
    pf_base = pos_base / neg_base if neg_base > 0 else 9.99
    
    pos_net = res_df[res_df["pnl_net_ftmo"] > 0]["pnl_net_ftmo"].sum()
    neg_net = abs(res_df[res_df["pnl_net_ftmo"] < 0]["pnl_net_ftmo"].sum())
    pf_net = pos_net / neg_net if neg_net > 0 else 9.99
    
    verdict = "NET_FTMO_FAILS"
    if pf_net > 1.5 and (total_r_net / len(res_df)) > 0.1: verdict = "NET_FTMO_STRONG"
    elif pf_net > 1.2 and total_r_net > 0: verdict = "NET_FTMO_SURVIVES"
    elif pf_net > 1.05 and total_r_net > 0: verdict = "NET_FTMO_FRAGILE"

    metrics = {
        "month": m_str, "sample": len(res_df), "total_R_base": round(total_r_base, 2), "PF_base": round(pf_base, 2),
        "total_R_net_FTMO": round(total_r_net, 2), "PF_net_FTMO": round(pf_net, 2),
        "verdict_net_FTMO": verdict, "batch_id": batch_id, "completed_at": datetime.utcnow().isoformat()
    }
    
    with open(os.path.join(batch_dir, f"PHASE56_MONTH_{m_clean}_METRICS.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 4. Checkpoint
    cp = load_checkpoint()
    cp["historical_progress"] = [p for p in cp["historical_progress"] if p["month"] != m_str]
    cp["historical_progress"].append(metrics)
    
    # Global summary
    all_canonic = [p for p in cp["historical_progress"] if "total_R_net_FTMO" in p]
    tr_net = sum(p["total_R_net_FTMO"] for p in all_canonic)
    t_samp = sum(p["sample"] for p in all_canonic)
    cp["summary"] = {
        "total_months": len(all_canonic), "total_sample": t_samp,
        "total_R_net_FTMO": round(tr_net, 2), "expectancy_net_FTMO": round(tr_net / t_samp, 4) if t_samp > 0 else 0
    }
    save_checkpoint(cp)
    
    update_live_status(batch_id, m_str, "FINISHED", last_completed=m_str, next_step="NEXT_MONTH")
    return True

def main():
    print("PHASE56J - STARTING QUEUE RUNNER")
    cp = load_checkpoint()
    
    processed_count = 0
    for batch in QUEUE:
        b_id = batch["id"]
        months = batch["months"]
        
        # Determine batch directory
        b_dir_name = f"batch_{months[0].replace('-','')}_{months[1].replace('-','')}"
        batch_dir = os.path.join(OUTPUT_BASE_DIR, b_dir_name)
        os.makedirs(batch_dir, exist_ok=True)
        
        for m_str in months:
            # Skip if already done
            if any(p["month"] == m_str for p in cp["historical_progress"] if p.get("total_R_net_FTMO") is not None):
                print(f"[SKIP] {m_str} already canonical.")
                continue
            
            success = process_month(b_id, m_str, batch_dir)
            if not success:
                print(f"[STOP] Safety or data error in {m_str}")
                return
            
            processed_count += 1
            if processed_count >= 50: # Safeguard
                print("[LIMIT] 50 months reached.")
                return

    print("PHASE56J - QUEUE COMPLETED")

if __name__ == "__main__":
    main()
