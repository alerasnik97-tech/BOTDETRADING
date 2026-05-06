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
    {"id": "batch_08", "month": "2016-06"},
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
    
    print(f"  [EXTRACTING] {year}-{month:02d}...")
    try:
        subprocess.run([sys.executable, EXTRACTOR_PATH, "--year", str(year), "--month", str(month), "--mode", "resume"], check=True)
        subprocess.run([sys.executable, EXTRACTOR_PATH, "--year", str(year), "--month", str(month), "--mode", "finalize"], check=True)
        subprocess.run([sys.executable, EXTRACTOR_PATH, "--year", str(year), "--month", str(month), "--mode", "validate"], check=True)
        if os.path.exists(tick_file):
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
        return {"status": "NO_TRADES"}
        
    tick_file = os.path.join(TICK_ROOT, f"EURUSD_ticks_{year}_{month:02d}.parquet")
    df_ticks = pd.read_parquet(tick_file)
    df_ticks["timestamp_utc"] = pd.to_datetime(df_ticks["timestamp_utc"], utc=True)
    df_ticks.sort_values("timestamp_utc", inplace=True)
    
    # Data Quality
    dq = {
        "rows": len(df_ticks),
        "first": df_ticks["timestamp_utc"].min().isoformat(),
        "last": df_ticks["timestamp_utc"].max().isoformat(),
        "bad_spread": len(df_ticks[df_ticks["bid"] > df_ticks["ask"]]),
        "neg_spread": len(df_ticks[df_ticks["ask"] - df_ticks["bid"] < 0])
    }
    pd.DataFrame([dq]).to_csv(os.path.join(batch_dir, f"PHASE56_BATCH_{year}{month:02d}_DATA_QUALITY.csv"), index=False)

    results = []
    for _, trade in df_month.iterrows():
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        sl_initial = trade["sl"]
        tp = trade["tp"]
        side = "LONG" if trade["type"].lower() == "buy" else "SHORT"
        risk_pips = trade["risk"] * 10000
        if risk_pips < 1.0: risk_pips = abs(entry_price - sl_initial) * 10000
        if risk_pips < 1.0: risk_pips = 10.0
        
        window = df_ticks[(df_ticks["timestamp_utc"] >= entry_time) & (df_ticks["timestamp_utc"] <= entry_time + timedelta(days=2))].copy()
        outcome = "TIME_EXIT"; exit_price = None; exit_time = None; be_active = False
        
        for _, tick in window.iterrows():
            t_utc = tick["timestamp_utc"]; t_ny = t_utc.astimezone(NY)
            bid = tick["bid"]; ask = tick["ask"]
            if t_ny.hour > 19 or (t_ny.hour == 19 and t_ny.minute >= 45):
                outcome = "TIME_EXIT"; exit_price = bid if side == "LONG" else ask; exit_time = t_utc; break
            if side == "LONG":
                if not be_active and (bid - entry_price) * 10000 >= 0.4 * risk_pips: be_active = True
                if bid >= tp: outcome = "TP"; exit_price = tp; exit_time = t_utc; break
                cur_sl = entry_price if be_active else sl_initial
                if bid <= cur_sl: outcome = "BE" if be_active else "SL"; exit_price = cur_sl; exit_time = t_utc; break
            else:
                if not be_active and (entry_price - ask) * 10000 >= 0.4 * risk_pips: be_active = True
                if ask <= tp: outcome = "TP"; exit_price = tp; exit_time = t_utc; break
                cur_sl = entry_price if be_active else sl_initial
                if ask >= cur_sl: outcome = "BE" if be_active else "SL"; exit_price = cur_sl; exit_time = t_utc; break
        
        if exit_price is None and not window.empty:
            last = window.iloc[-1]; exit_price = last["bid"] if side == "LONG" else last["ask"]; exit_time = last["timestamp_utc"]
            
        pnl_r = ((exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)) * 10000 / risk_pips if exit_price else 0
        if outcome == "TP": pnl_r = 1.4
        elif outcome == "SL": pnl_r = -1.0
        elif outcome == "BE": pnl_r = 0.0
        
        comm_r = 0.5 / risk_pips
        results.append({
            "trade_id": trade.get("trade_id", "N/A"), "entry_time": entry_time.isoformat(), "outcome": outcome,
            "pnl_r_base": round(pnl_r, 4), "commission_r": round(comm_r, 4), "pnl_r_net": round(pnl_r - comm_r, 4)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(batch_dir, f"PHASE56_BATCH_{year}{month:02d}_TRADE_LEVEL.csv"), index=False)
    
    pos = res_df[res_df["pnl_r_base"] > 0]["pnl_r_base"].sum()
    neg = abs(res_df[res_df["pnl_r_base"] < 0]["pnl_r_base"].sum())
    pf_base = pos / neg if neg > 0 else pos
    
    pos_n = res_df[res_df["pnl_r_net"] > 0]["pnl_r_net"].sum()
    neg_n = abs(res_df[res_df["pnl_r_net"] < 0]["pnl_r_net"].sum())
    pf_net = pos_n / neg_n if neg_n > 0 else pos_n
    
    metrics = {
        "month": f"{year}-{month:02d}", "sample": len(res_df), "PF_base": round(pf_base, 2),
        "expectancy_base": round(res_df["pnl_r_base"].mean(), 4), "total_R_base": round(res_df["pnl_r_base"].sum(), 2),
        "PF_net_FTMO": round(pf_net, 2), "expectancy_net_FTMO": round(res_df["pnl_r_net"].mean(), 4),
        "total_R_net_FTMO": round(res_df["pnl_r_net"].sum(), 2), "avg_commission_R": round(res_df["commission_r"].mean(), 4)
    }
    metrics["verdict_net_FTMO"] = "NET_FTMO_STRONG" if pf_net > 1.5 and metrics["expectancy_net_FTMO"] > 0.1 else "NET_FTMO_SURVIVES" if pf_net > 1.2 else "NET_FTMO_FRAGILE" if pf_net > 1.05 else "NET_FTMO_FAILS"
    
    # Stress
    res_df["pnl_01"] = res_df["pnl_r_net"] - 0.1
    res_df["pnl_02"] = res_df["pnl_r_net"] - 0.2
    stress = {
        "PF_0.1R": round(res_df[res_df["pnl_01"] > 0]["pnl_01"].sum() / abs(res_df[res_df["pnl_01"] < 0]["pnl_01"].sum()), 2) if len(res_df[res_df["pnl_01"] < 0]) > 0 else 0,
        "PF_0.2R": round(res_df[res_df["pnl_02"] > 0]["pnl_02"].sum() / abs(res_df[res_df["pnl_02"] < 0]["pnl_02"].sum()), 2) if len(res_df[res_df["pnl_02"] < 0]) > 0 else 0
    }
    with open(os.path.join(batch_dir, f"PHASE56_BATCH_{year}{month:02d}_STRESS.json"), "w") as f: json.dump(stress, f, indent=4)
    with open(os.path.join(batch_dir, f"PHASE56_BATCH_{year}{month:02d}_FTMO_NET.json"), "w") as f: json.dump(metrics, f, indent=4)
    
    return metrics

def main():
    cp = load_checkpoint()
    for b in QUEUE:
        m_str = b["month"]
        if any(p["month"] == m_str for p in cp["historical_progress"]):
            print(f"[SKIP] {m_str}")
            continue
        
        y, m = map(int, m_str.split("-"))
        b_dir = os.path.join(OUTPUT_ROOT, f"batch_{y}{m:02d}")
        os.makedirs(b_dir, exist_ok=True)
        
        ok, msg = ensure_tick_data(y, m)
        if not ok: continue
        
        metrics = run_replay(y, m, b_dir)
        if metrics.get("status") == "NO_TRADES":
            metrics = {"month": m_str, "sample": 0, "PF_base": 0, "total_R_net_FTMO": 0, "verdict_net_FTMO": "NO_TRADES"}
        
        metrics["batch_id"] = b["id"]
        metrics["completed_at"] = datetime.now().isoformat()
        cp["historical_progress"].append(metrics)
        cp[b["id"]] = {"month": m_str, "status": "COMPLETED", "timestamp": metrics["completed_at"]}
        save_checkpoint(cp)
        print(f"  [DONE] {m_str}: Net_R={metrics.get('total_R_net_FTMO')}")

    # Summary Update
    all_p = cp["historical_progress"]
    cp["summary"] = {
        "total_months": len(all_p), "total_sample": sum(p.get("sample", 0) for p in all_p),
        "total_R_net_FTMO": round(sum(p.get("total_R_net_FTMO", 0) for p in all_p), 2),
        "expectancy_net_FTMO": round(np.mean([p.get("expectancy_net_FTMO", 0) for p in all_p if p.get("sample", 0) > 0]), 4)
    }
    save_checkpoint(cp)

if __name__ == "__main__":
    main()
