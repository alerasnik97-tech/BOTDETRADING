
import pandas as pd
from pathlib import Path

def test_load():
    p = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_m3\EURUSD_M3_BID_2020_2026.csv"
    print(f"Loading {p}...")
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows.")

if __name__ == "__main__":
    test_load()
