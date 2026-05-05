import os
import sys
import json
import pandas as pd
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Setup paths
ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
SRC = LAB / "src"
OUT = LAB / "outputs" / "phase41_manipulante_hybrid_replay_forward_audit"
MANIPULANTE = ROOT / "MANIPULANTE"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase37_ftmo_trial_support import NY
from phase37d_manipulante_live_signal_engine import generate_phase25_signals_from_m3
import phase37x_session_lifecycle as lifecycle

UTC = timezone.utc

def load_certified_m3(limit=None):
    manifest_path = LAB / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    df = pd.read_csv(manifest["bid_path"], nrows=limit)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert(NY)
    rename = {col: f"{col}_bid" for col in ["open", "high", "low", "close"]}
    df = df.rename(columns=rename)
    return df

def run_replay(start_date, end_date, mode="recent"):
    print(f"[REPLAY] Mode: {mode} | Range: {start_date} to {end_date}")
    
    data_start = pd.to_datetime(start_date, utc=True) - timedelta(days=10) 
    full_df = load_certified_m3()
    df = full_df[(full_df["timestamp"] >= data_start) & (full_df["timestamp"] <= pd.to_datetime(end_date, utc=True))].copy()
    
    if df.empty:
        print("[ERROR] No data found for range.")
        return []

    print("[REPLAY] Detecting signals using phase37d engine...")
    signals = generate_phase25_signals_from_m3(df)
    
    if signals.empty:
        print("[REPLAY] No signals detected in range.")
        return []

    # Filter signals to the requested range
    replay_signals = signals[(signals["choch_time"] >= pd.to_datetime(start_date).tz_localize(NY)) & 
                             (signals["choch_time"] <= pd.to_datetime(end_date).tz_localize(NY))].copy()
    
    results = []
    for _, sig in replay_signals.iterrows():
        # Map columns based on First3MChochDetector
        entry_price = sig["entry_price"]
        side = sig["direction"] # Was 'direction' in detector
        sl_price = sig["sl_price"]
        risk_pips = abs(entry_price - sl_price)
        
        tp_price = entry_price + (risk_pips * 1.4) if side.upper() == "LONG" else entry_price - (risk_pips * 1.4)
        be_trigger_price = entry_price + (risk_pips * 0.4) if side.upper() == "LONG" else entry_price - (risk_pips * 0.4)
        
        # Simulate life of trade
        trade_data = full_df[full_df["timestamp"] > sig["choch_time"]].head(1000)
        
        outcome = "UNKNOWN"
        exit_time = None
        exit_price = None
        be_hit = False
        
        for _, bar in trade_data.iterrows():
            if not be_hit:
                if side.upper() == "LONG" and bar["high_bid"] >= be_trigger_price:
                    be_hit = True
                elif side.upper() == "SHORT" and bar["low_bid"] <= be_trigger_price:
                    be_hit = True
            
            if side.upper() == "LONG" and bar["high_bid"] >= tp_price:
                outcome = "TP"
                exit_price = tp_price
                exit_time = bar["timestamp_ny"]
                break
            elif side.upper() == "SHORT" and bar["low_bid"] <= tp_price:
                outcome = "TP"
                exit_price = tp_price
                exit_time = bar["timestamp_ny"]
                break
                
            if side.upper() == "LONG" and bar["low_bid"] <= sl_price:
                outcome = "BE" if be_hit else "SL"
                exit_price = entry_price if be_hit else sl_price
                exit_time = bar["timestamp_ny"]
                break
            elif side.upper() == "SHORT" and bar["high_bid"] >= sl_price:
                outcome = "BE" if be_hit else "SL"
                exit_price = entry_price if be_hit else sl_price
                exit_time = bar["timestamp_ny"]
                break
            
            # Friday hard close
            if bar["timestamp_ny"].weekday() == 4 and bar["timestamp_ny"].hour == 16 and bar["timestamp_ny"].minute >= 55:
                outcome = "WEEKEND_CLOSE"
                exit_price = bar["close_bid"]
                exit_time = bar["timestamp_ny"]
                break

            # Daily close
            if bar["timestamp_ny"].hour == 19 and bar["timestamp_ny"].minute >= 45:
                outcome = "DAILY_CLOSE"
                exit_price = bar["close_bid"]
                exit_time = bar["timestamp_ny"]
                break
        
        r_gross = 0
        if outcome == "TP": r_gross = 1.4
        elif outcome == "SL": r_gross = -1.0
        elif outcome == "BE": r_gross = 0.0
        elif outcome in ["DAILY_CLOSE", "WEEKEND_CLOSE"]:
            r_gross = (exit_price - entry_price) / risk_pips if side.upper() == "LONG" else (entry_price - exit_price) / risk_pips
            
        results.append({
            "timestamp_ny": sig["choch_time"].isoformat(),
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "outcome": outcome,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "exit_price": exit_price,
            "r_gross": round(r_gross, 4),
            "be_hit": be_hit
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="recent")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-30")
    args = parser.parse_args()
    
    os.makedirs(OUT / "decisions_like_live", exist_ok=True)
    
    if args.mode == "recent":
        res = run_replay(args.start, args.end, mode="RECENT")
        if res: pd.DataFrame(res).to_csv(OUT / "decisions_like_live" / "recent_decisions.csv", index=False)
    elif args.mode == "sample_windows":
        windows = [
            ("2025-02-01", "2025-02-28", "critical_2025_02"),
            ("2024-06-01", "2024-06-30", "best_2024_06")
        ]
        for s, e, lbl in windows:
            res = run_replay(s, e, mode=lbl)
            if res: pd.DataFrame(res).to_csv(OUT / "decisions_like_live" / f"{lbl}_decisions.csv", index=False)
    
    print("[DONE] Replay finished.")

if __name__ == "__main__":
    main()
