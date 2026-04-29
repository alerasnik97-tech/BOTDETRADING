import os
import pandas as pd

def main():
    root = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
    raw_dir = os.path.join(root, "data_intake_2015_2019", "raw_m1", "2015")
    out_dir = os.path.join(root, "BOT_V2_DAYTIME_LAB", "data", "processed_2015_2019", "eurusd_m1_certified_candidate", "2015_01")
    os.makedirs(out_dir, exist_ok=True)

    bid_path = os.path.join(raw_dir, "EURUSD_M1_BID.csv")
    ask_path = os.path.join(raw_dir, "EURUSD_M1_ASK.csv")

    print("Loading BID data...")
    df_bid = pd.read_csv(bid_path)
    df_bid['timestamp'] = pd.to_datetime(df_bid['timestamp'])
    
    print("Loading ASK data...")
    df_ask = pd.read_csv(ask_path)
    df_ask['timestamp'] = pd.to_datetime(df_ask['timestamp'])

    # Pilot: 2015-01
    print("Filtering Jan 2015...")
    df_bid = df_bid[df_bid['timestamp'].dt.month == 1]
    df_ask = df_ask[df_ask['timestamp'].dt.month == 1]

    print("Merging...")
    df = pd.merge(df_bid, df_ask, on='timestamp', suffixes=('_bid', '_ask'))
    
    # Format M1 Canónico
    df = df.rename(columns={
        'open_bid': 'bid_open', 'high_bid': 'bid_high', 'low_bid': 'bid_low', 'close_bid': 'bid_close',
        'open_ask': 'ask_open', 'high_ask': 'ask_high', 'low_ask': 'ask_low', 'close_ask': 'ask_close'
    })
    
    df['spread_open'] = df['ask_open'] - df['bid_open']
    df['spread_high'] = df['ask_high'] - df['bid_high']
    df['spread_low'] = df['ask_low'] - df['bid_low']
    df['spread_close'] = df['ask_close'] - df['bid_close']
    df['source'] = 'Dukascopy'
    df['quality_flags'] = 0

    out_path = os.path.join(out_dir, "EURUSD_M1_2015_01.csv")
    print(f"Saving to {out_path}...")
    df.to_csv(out_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
