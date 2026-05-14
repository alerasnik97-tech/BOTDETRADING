import pandas as pd
from pathlib import Path

vault_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD\monthly")
df = pd.read_parquet(vault_path / "EURUSD_ticks_2022_05.parquet")
print("Tick Head:")
print(df.head())

bars_15m = df["bid"].resample("15min").ohlc().dropna()
print("\nBars 15m Head:")
print(bars_15m.head(20))

for ts in bars_15m.index[:100]:
    if ts.hour == 3:
        print(f"Found London Hour: {ts}")
