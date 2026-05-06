import os
import pandas as pd

def main():
    root = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
    in_path = os.path.join(root, "BOT_V2_DAYTIME_LAB", "data", "processed_2015_2019", "eurusd_m1_certified_candidate", "2015_01", "EURUSD_M1_2015_01.csv")
    out_dir = os.path.join(root, "BOT_V2_DAYTIME_LAB", "data", "processed_2015_2019", "eurusd_m3_from_m1", "2015_01")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating M3 from {in_path}...")
    df = pd.read_csv(in_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Aggregation logic
    resampler = df.resample('3min', closed='left', label='right')
    
    m3 = pd.DataFrame()
    m3['bid_open'] = resampler['bid_open'].first()
    m3['bid_high'] = resampler['bid_high'].max()
    m3['bid_low'] = resampler['bid_low'].min()
    m3['bid_close'] = resampler['bid_close'].last()
    
    m3['ask_open'] = resampler['ask_open'].first()
    m3['ask_high'] = resampler['ask_high'].max()
    m3['ask_low'] = resampler['ask_low'].min()
    m3['ask_close'] = resampler['ask_close'].last()
    
    m3 = m3.shift(1).dropna()
    m3 = m3.reset_index()

    out_path = os.path.join(out_dir, "EURUSD_M3_2015_01.csv")
    m3.to_csv(out_path, index=False)
    print(f"Saved {len(m3)} rows to {out_path}")

if __name__ == "__main__":
    main()
