
import os
from pathlib import Path

def preflight():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    staging_dir = root / "data_intake_2015_2019"
    prepared_dir = staging_dir / "prepared"
    news_file = staging_dir / "news" / "news_eurusd_2015_2019_fortress_candidate.csv"
    
    price_files = [
        "EURUSD_M5_BID.csv", "EURUSD_M5_ASK.csv", "EURUSD_M5_MID.csv", "EURUSD_M5_SPREAD.csv",
        "EURUSD_H1_BID.csv", "EURUSD_H1_ASK.csv", "EURUSD_H1_MID.csv", "EURUSD_H1_SPREAD.csv"
    ]
    
    missing = []
    for f in price_files:
        if not (prepared_dir / f).exists():
            missing.append(f"Price: {f}")
    
    if not news_file.exists():
        missing.append(f"News: {news_file.name}")
        
    if missing:
        print("FAIL-CLOSED: Missing files:")
        for m in missing:
            print(f" - {m}")
        return False
    
    print("PREFLIGHT: All files exist.")
    return True

if __name__ == "__main__":
    if preflight():
        print("PREFLIGHT_SUCCESS")
    else:
        print("PREFLIGHT_FAIL")
