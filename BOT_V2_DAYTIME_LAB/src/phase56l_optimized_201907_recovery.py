import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys

# CONFIGURATION
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
TICK_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_2019_07.parquet"
CHECKPOINT_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\PHASE56_FULL_HISTORICAL_CHECKPOINT.json"
OUTPUT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\batch_201907_201908"
LIVE_STATUS_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\PHASE56_LIVE_STATUS.txt"

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

def update_live_status(phase, progress=""):
    with open(LIVE_STATUS_PATH, "w") as f:
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"BATCH_ACTUAL: BATCH_27_RECOVERY\n")
        f.write(f"MES_ACTUAL: 2019-07\n")
        f.write(f"FASE_ACTUAL: {phase}\n")
        f.write(f"PROGRESS: {progress}\n")
        f.write(f"ULTIMO_MES: 2019-06\n")
        f.write(f"PROXIMO_PASO: RECOVERY_IN_PROGRESS\n")

def load_checkpoint():
    with open(CHECKPOINT_PATH, 'r') as f:
        return json.load(f)

def save_checkpoint(cp):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(cp, f, indent=4)

def main():
    update_live_status("PREFLIGHT")
    print("[PHASE56L] Starting optimized recovery for 2019-07...")

    # 1. Preflight
    if not os.path.exists(TICK_PATH):
        print(f"[ERROR] Parquet missing: {TICK_PATH}")
        sys.exit(1)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Data Quality & Load
    update_live_status("LOADING_DATA")
    print("[PHASE56L] Loading Parquet (this might take a moment)...")
    df_ticks = pd.read_parquet(TICK_PATH)
    df_ticks["timestamp_utc"] = pd.to_datetime(df_ticks["timestamp_utc"], utc=True)
    df_ticks.sort_values("timestamp_utc", inplace=True)
    
    dq = {
        "rows": len(df_ticks),
        "first_ts": df_ticks["timestamp_utc"].min().isoformat(),
        "last_ts": df_ticks["timestamp_utc"].max().isoformat(),
        "nulls": df_ticks.isnull().sum().to_dict(),
        "spread_negative": int((df_ticks["ask"] - df_ticks["bid"] < 0).sum())
    }
    with open(os.path.join(OUTPUT_DIR, "PHASE56_MONTH_201907_DATA_QUALITY.json"), "w") as f:
        json.dump(dq, f, indent=4)

    # 3. Trades
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)
    df_month = df_raw[(df_raw["entry_time"].dt.year == 2019) & (df_raw["entry_time"].dt.month == 7)].copy()
    
    print(f"[PHASE56L] Processing {len(df_month)} trades...")
    
    results = []
    trade_level_path = os.path.join(OUTPUT_DIR, "PHASE56_MONTH_201907_TRADE_LEVEL.csv")

    for i, (_, trade) in enumerate(df_month.iterrows()):
        trade_id = trade.get("trade_id", f"T_{i}")
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        sl = trade["sl"]
        tp = trade["tp"]
        side = "LONG" if trade["type"].lower() == "buy" else "SHORT"
        risk_pips = abs(entry_price - sl) * 10000
        commission_r = 0.5 / risk_pips if risk_pips > 0 else 0
        
        # Optimization: Slice window (max 48h or until 19:45 NY of next day)
        end_window = entry_time + timedelta(days=2)
        window = df_ticks[(df_ticks["timestamp_utc"] >= entry_time) & (df_ticks["timestamp_utc"] <= end_window)].copy()
        
        outcome = "TIME_EXIT"
        exit_price, exit_time = None, None
        be_active = False
        auditable = "YES"
        reason = "NONE"

        if window.empty:
            auditable = "NO"
            reason = "NO_TICKS_IN_WINDOW"
            outcome = "FAILED_NO_TICKS"
            pnl_net_ftmo = np.nan
        else:
            # Convert to values for speed
            ts_values = window["timestamp_utc"].values
            bid_values = window["bid"].values
            ask_values = window["ask"].values
            
            # Replay Loop (Manual but slightly faster with values)
            for j in range(len(window)):
                t_utc = pd.Timestamp(ts_values[j], tz='UTC')
                t_ny = t_utc.astimezone(NY)
                bid, ask = bid_values[j], ask_values[j]
                
                # Cierre 19:45 NY
                if t_ny.hour > 19 or (t_ny.hour == 19 and t_ny.minute >= 45):
                    outcome = "TIME_EXIT"
                    exit_price = bid if side == "LONG" else ask
                    exit_time = t_utc
                    break
                
                # Gestion
                if side == "LONG":
                    # BE Trigger (+0.4R)
                    if not be_active and (bid - entry_price) * 10000 >= 0.4 * risk_pips:
                        be_active = True
                    
                    # Check TP first (matching Phase 56J order)
                    if bid >= tp:
                        outcome = "TP"
                        exit_price = tp
                        exit_time = t_utc
                        break

                    # SL / BE Stop
                    stop_lvl = entry_price if be_active else sl
                    if bid <= stop_lvl:
                        outcome = "BE" if be_active else "SL"
                        exit_price = stop_lvl
                        exit_time = t_utc
                        break
                else: # SHORT
                    # BE Trigger
                    if not be_active and (entry_price - ask) * 10000 >= 0.4 * risk_pips:
                        be_active = True
                    
                    # Check TP first
                    if ask <= tp:
                        outcome = "TP"
                        exit_price = tp
                        exit_time = t_utc
                        break

                    # SL / BE Stop
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

        res = {
            "trade_id": trade_id,
            "entry_time": entry_time.isoformat(),
            "side": side,
            "outcome": outcome,
            "pnl_net_ftmo": round(pnl_net_ftmo, 4) if not np.isnan(pnl_net_ftmo) else "NaN",
            "auditable": auditable,
            "reason": reason
        }
        results.append(res)
        
        # Incremental Save
        if (i + 1) % 5 == 0 or (i + 1) == len(df_month):
            pd.DataFrame(results).to_csv(trade_level_path, index=False)
            update_live_status("REPLAY", f"{i+1}/{len(df_month)}")
            print(f"[PHASE56L] Progress: {i+1}/{len(df_month)}")

    # 4. Metrics
    update_live_status("METRICS")
    df_res = pd.DataFrame(results)
    df_res["pnl_net_ftmo"] = pd.to_numeric(df_res["pnl_net_ftmo"], errors='coerce')
    
    auditables = df_res[df_res["auditable"] == "YES"]
    total_r = auditables["pnl_net_ftmo"].sum()
    sample = len(df_res)
    
    metrics = {
        "month": "2019-07",
        "sample_total": sample,
        "auditables": len(auditables),
        "non_auditables": sample - len(auditables),
        "total_R_net_FTMO": round(total_r, 4),
        "expectancy_net_FTMO": round(total_r / len(auditables), 4) if len(auditables) > 0 else 0,
        "PF_net_FTMO": round(auditables[auditables["pnl_net_ftmo"] > 0]["pnl_net_ftmo"].sum() / abs(auditables[auditables["pnl_net_ftmo"] < 0]["pnl_net_ftmo"].sum()), 2) if auditables[auditables["pnl_net_ftmo"] < 0]["pnl_net_ftmo"].sum() != 0 else 99,
        "verdict_net_FTMO": "NET_FTMO_STRONG" if total_r > 5 else "NET_FTMO_SURVIVES" if total_r > 0 else "NET_FTMO_FAILS"
    }
    
    with open(os.path.join(OUTPUT_DIR, "PHASE56_MONTH_201907_METRICS.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # 5. Checkpoint Update
    update_live_status("CHECKPOINT")
    cp = load_checkpoint()
    
    # Avoid duplicates
    cp["historical_progress"] = [m for m in cp["historical_progress"] if m["month"] != "2019-07"]
    
    cp_entry = {
        "month": "2019-07",
        "sample_total": metrics["sample_total"],
        "auditables": metrics["auditables"],
        "non_auditables": metrics["non_auditables"],
        "total_R_net_FTMO": metrics["total_R_net_FTMO"],
        "PF_net_FTMO": metrics["PF_net_FTMO"],
        "expectancy_net_FTMO": metrics["expectancy_net_FTMO"],
        "verdict_net_FTMO": metrics["verdict_net_FTMO"],
        "batch_id": "batch_201907_201908",
        "replay_status": "FORENSIC_COMPLETE",
        "completed_at": datetime.utcnow().isoformat()
    }
    cp["historical_progress"].append(cp_entry)
    
    # Re-calculate summary
    all_net = [m.get("total_R_net_FTMO", 0) for m in cp["historical_progress"] if "total_R_net_FTMO" in m]
    all_sample = [m.get("sample_total", m.get("sample", 0)) for m in cp["historical_progress"]]
    
    cp["summary"] = {
        "total_months": len(cp["historical_progress"]),
        "total_sample": sum(all_sample),
        "total_R_net_FTMO": round(sum(all_net), 4),
        "expectancy_net_FTMO": round(sum(all_net) / sum(all_sample), 4) if sum(all_sample) > 0 else 0
    }
    
    save_checkpoint(cp)
    
    # Recovery Report
    report = {
        "status": "PHASE56L_201907_RECOVERY_COMPLETED",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics
    }
    with open(os.path.join(OUTPUT_DIR, "PHASE56L_201907_RECOVERY_REPORT.json"), "w") as f:
        json.dump(report, f, indent=4)

    update_live_status("COMPLETED", "19/19")
    print("[PHASE56L] Recovery completed successfully.")

if __name__ == "__main__":
    main()
