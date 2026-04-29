import os
import pandas as pd
import numpy as np

def main():
    root = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
    in_path = os.path.join(root, "BOT_V2_DAYTIME_LAB", "data", "processed_2015_2019", "eurusd_m1_certified_candidate", "2015_01", "EURUSD_M1_2015_01.csv")
    out_dir = os.path.join(root, "BOT_V2_DAYTIME_LAB", "outputs", "phase26b_data_engineering", "m1_quality_pilot")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Auditing {in_path}...")
    df = pd.read_csv(in_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Basic stats
    total_rows = len(df)
    
    # Gaps
    df = df.sort_values('timestamp')
    df['diff'] = df['timestamp'].diff().dt.total_seconds()
    gaps = df[df['diff'] > 60]
    
    # Spread
    neg_spread = df[df['spread_close'] < 0]
    
    # Results
    summary = {
        "total_rows": total_rows,
        "gaps_count": len(gaps),
        "neg_spread_count": len(neg_spread),
        "min_timestamp": str(df['timestamp'].min()),
        "max_timestamp": str(df['timestamp'].max())
    }
    
    import json
    with open(os.path.join(out_dir, "phase26b_m1_quality_pilot_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    gaps.to_csv(os.path.join(out_dir, "phase26b_m1_quality_pilot_gap_report.csv"), index=False)
    
    print("Audit summary:")
    print(summary)
    print("Done.")

if __name__ == "__main__":
    main()
