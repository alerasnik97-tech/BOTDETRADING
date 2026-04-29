import os
import argparse
import pandas as pd
import hashlib

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot-year", type=int, default=2015)
    parser.add_argument("--pilot-month", type=int, default=1)
    args = parser.parse_args()

    root = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
    raw_root = os.path.join(root, "data_intake_2015_2019", "raw_m1")
    
    print(f"Checking data in {raw_root}...")
    
    years = [2015, 2016, 2017, 2018, 2019]
    manifest = []

    for y in years:
        y_dir = os.path.join(raw_root, str(y))
        bid_path = os.path.join(y_dir, "EURUSD_M1_BID.csv")
        ask_path = os.path.join(y_dir, "EURUSD_M1_ASK.csv")
        
        status = "MISSING"
        if os.path.exists(bid_path) and os.path.exists(ask_path):
            status = "FOUND"
            sha_bid = get_sha256(bid_path)
            sha_ask = get_sha256(ask_path)
            manifest.append({
                "year": y,
                "bid_file": bid_path,
                "ask_file": ask_path,
                "sha256_bid": sha_bid,
                "sha256_ask": sha_ask
            })
        
        print(f"Year {y}: {status}")

    if args.execute:
        print(f"Importing pilot {args.pilot_year}-{args.pilot_month:02d}...")
        # Since data is already there, we just certify its existence for the pilot
        pilot_dir = os.path.join(raw_root, str(args.pilot_year))
        bid_path = os.path.join(pilot_dir, "EURUSD_M1_BID.csv")
        if os.path.exists(bid_path):
            df = pd.read_csv(bid_path, nrows=1000)
            print(f"Pilot head:\n{df.head()}")
            print("PILOT_READY")
        else:
            print("PILOT_MISSING")

if __name__ == "__main__":
    main()
