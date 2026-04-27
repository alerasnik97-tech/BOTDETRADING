from pathlib import Path
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.config import DEFAULT_PAIR, DEFAULT_DATA_DIRS
from research_lab.strategies.common import is_in_session

def debug():
    bundle = load_backtest_data_bundle(
        DEFAULT_PAIR,
        [Path(d) for d in DEFAULT_DATA_DIRS],
        "2020-01-01",
        "2020-02-01",
        "normal_mode"
    )
    df = bundle.frame
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Check session columns
    cols = ["prev_day_high", "prev_day_low", "high", "low", "close", "day_running_high"]
    print("\nSample Data (First 10 rows):")
    print(df[cols].head(10))
    
    # Check if there are ANY rows where high > prev_day_high in the PM window
    df["is_pm"] = [is_in_session(t, "light_fixed") for t in df.index]
    sweeps_pm = df[(df["high"] > df["prev_day_high"]) & df["is_pm"]]
    print(f"\nRows where high > prev_day_high (PM ONLY): {len(sweeps_pm)}")
    
    # Simulate Strategy Logic
    sweep_dist = 0.5 * 0.0001
    results = []
    for i in range(1, len(df)):
        pdh = df["prev_day_high"].iat[i]
        high_curr = df["high"].iat[i]
        close_curr = df["close"].iat[i]
        close_prev = df["close"].iat[i-1]
        
        if not is_in_session(df.index[i], "light_fixed"):
            continue
            
        trigger_a = (high_curr > pdh + sweep_dist) and (close_curr < pdh)
        trigger_b = (close_prev >= pdh) and (close_curr < pdh)
        
        if trigger_a or trigger_b:
            results.append(df.index[i])
            
    print(f"\nSimulated Signal Triggers (PM ONLY): {len(results)}")
    if results:
        print(results[:10])

if __name__ == "__main__":
    debug()
