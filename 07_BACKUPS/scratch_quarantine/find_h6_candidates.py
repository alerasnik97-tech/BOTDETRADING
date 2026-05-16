import pandas as pd
from pathlib import Path

def find_candidates():
    search_dirs = [
        Path("results/DEEP_RESEARCH_CAMPAIGN_2.0/H6_SILVER_BULLET_HYBRID"),
        Path("results/DEEP_RESEARCH_CAMPAIGN_2.0_OOS/H6_SILVER_BULLET_HYBRID")
    ]
    
    existing_dates = [
        "2021-04-01", "2021-11-09", "2022-05-18", "2022-08-25", "2022-11-09",
        "2023-01-10", "2023-01-31", "2023-02-14", "2023-03-29", "2023-03-31",
        "2024-06-06", "2024-09-06", "2025-01-29", "2025-03-26", "2025-10-01"
    ]
    
    candidates = []
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for trades_file in base_dir.glob("*/trades.csv"):
            df = pd.read_csv(trades_file)
            for _, row in df.iterrows():
                signal_time = pd.Timestamp(row["signal_time_ny"])
                if signal_time.hour == 10:
                    date_str = signal_time.strftime("%Y-%m-%d")
                    if date_str not in existing_dates:
                        candidates.append(date_str)
    
    return sorted(list(set(candidates)))

if __name__ == "__main__":
    print(find_candidates())
