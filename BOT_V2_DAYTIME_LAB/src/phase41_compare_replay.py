import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUT = LAB / "outputs" / "phase41_manipulante_hybrid_replay_forward_audit"
EXPECTED_PATH = LAB / "outputs" / "phase27_full_historical_validation_2015_2026" / "validation_2015_2026_full" / "phase27_2015_2026_trades.csv"
REPLAY_PATH = OUT / "decisions_like_live" / "recent_decisions.csv"

def compare():
    if not REPLAY_PATH.exists():
        print("[ERROR] Replay file missing.")
        return

    # Load expected (Phase 27)
    # Header: side,choch_time,entry_bid,sl_bid,tp_bid,risk_pips,outcome,be_hit,exit_bid,exit_time
    expected = pd.read_csv(EXPECTED_PATH, names=["side","choch_time","entry_bid","sl_bid","tp_bid","risk_pips","outcome","be_hit","exit_bid","exit_time"], header=0)
    expected["choch_time"] = pd.to_datetime(expected["choch_time"], utc=True)
    
    # Load replay
    replay = pd.read_csv(REPLAY_PATH)
    replay["timestamp_ny"] = pd.to_datetime(replay["timestamp_ny"], utc=True)
    
    # Filter expected to the same range as replay
    start_date = replay["timestamp_ny"].min()
    end_date = replay["timestamp_ny"].max()
    expected_filtered = expected[(expected["choch_time"] >= start_date) & (expected["choch_time"] <= end_date)].copy()
    
    print(f"[COMPARE] Expected: {len(expected_filtered)} | Replay: {len(replay)}")
    
    # Merge for comparison
    # We round timestamps to minutes to avoid slight differences
    expected_filtered["time_min"] = expected_filtered["choch_time"].dt.floor("min")
    replay["time_min"] = replay["timestamp_ny"].dt.floor("min")
    
    merged = pd.merge(expected_filtered, replay, left_on="time_min", right_on="time_min", how="outer", suffixes=("_exp", "_rep"))
    
    # Missing trades (in expected but not in replay)
    missing = merged[merged["timestamp_ny"].isna()].copy()
    
    # Extra trades (in replay but not in expected)
    extra = merged[merged["choch_time"].isna()].copy()
    
    # Mismatches (both present but different outcome)
    mismatches = merged[merged["timestamp_ny"].notna() & merged["choch_time"].notna() & (merged["outcome_exp"] != merged["outcome_rep"])].copy()
    
    # Save results
    os.makedirs(OUT / "comparison", exist_ok=True)
    merged.to_csv(OUT / "comparison" / "phase41_expected_vs_replay_trades.csv", index=False)
    missing.to_csv(OUT / "comparison" / "phase41_missing_trades.csv", index=False)
    extra.to_csv(OUT / "comparison" / "phase41_extra_trades.csv", index=False)
    mismatches.to_csv(OUT / "comparison" / "phase41_outcome_mismatches.csv", index=False)
    
    print(f"[DONE] Comparison finished. Missing: {len(missing)}, Extra: {len(extra)}, Mismatches: {len(mismatches)}")

import os
if __name__ == "__main__":
    compare()
