import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = PROJECT_ROOT / "data_intake_2015_2019" / "prepared"

def merge_prices(timeframe):
    print(f"Merging {timeframe} prices...")
    bid = pd.read_csv(PREPARED_DIR / f"EURUSD_{timeframe}_BID.csv", index_col=0, parse_dates=True)
    ask = pd.read_csv(PREPARED_DIR / f"EURUSD_{timeframe}_ASK.csv", index_col=0, parse_dates=True)
    mid_df = pd.read_csv(PREPARED_DIR / f"EURUSD_{timeframe}_MID.csv", index_col=0, parse_dates=True)
    spread_df = pd.read_csv(PREPARED_DIR / f"EURUSD_{timeframe}_SPREAD.csv", index_col=0, parse_dates=True)
    
    # Renombrar columnas de BID y ASK
    bid.columns = [f"Open_BID", "High_BID", "Low_BID", "Close_BID", "Volume_BID"]
    ask.columns = [f"Open_ASK", "High_ASK", "Low_ASK", "Close_ASK", "Volume_ASK"]
    
    # Seleccionar MID (Close) y SPREAD (spread_mean)
    mid = mid_df[["close"]].rename(columns={"close": "MID"})
    spread = spread_df[["spread_mean"]].rename(columns={"spread_mean": "SPREAD"})
    
    # Combinar
    df = pd.concat([bid, ask, mid, spread], axis=1).dropna()
    
    output_file = PREPARED_DIR / f"EURUSD_{timeframe}_2015_2019_BID_ASK_MID_SPREAD.csv"
    df.to_csv(output_file)
    print(f"Saved merged file to {output_file} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    merge_prices("M5")
    merge_prices("H1")
