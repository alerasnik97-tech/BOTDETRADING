import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data_intake_2015_2019" / "raw_m1"
MERGED_DIR = PROJECT_ROOT / "data_intake_2015_2019" / "raw_m1_merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

def merge_years():
    for side in ("BID", "ASK"):
        all_frames = []
        for year in range(2015, 2020):
            path = RAW_DIR / str(year) / f"EURUSD_M1_{side}.csv"
            if path.exists():
                print(f"Loading {path}...")
                all_frames.append(pd.read_csv(path))
        
        merged = pd.concat(all_frames).sort_values("timestamp").drop_duplicates("timestamp")
        output_file = MERGED_DIR / f"EURUSD_M1_{side}.csv"
        merged.to_csv(output_file, index=False)
        print(f"Saved merged {side} to {output_file}")

if __name__ == "__main__":
    merge_years()
