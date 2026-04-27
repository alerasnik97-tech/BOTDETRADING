import os
import pandas as pd
from pathlib import Path

root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
for path in root.rglob("*.csv"):
    try:
        df = pd.read_csv(path, nrows=1)
        if not df.empty:
            first_val = str(df.iloc[0, 0])
            if "2015" in first_val or "2016" in first_val or "2017" in first_val:
                print(f"FOUND {path}: {first_val}")
    except:
        pass
