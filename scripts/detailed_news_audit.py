import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_FILE = PROJECT_ROOT / "data" / "news_eurusd_v2_utc_audit.csv"

MAPPING = {
    "NFP": ["non-farm employment change", "nonfarm payrolls"],
    "CPI": ["cpi m/m", "cpi y/y", "core cpi m/m"],
    "FOMC": ["federal funds rate", "fomc statement", "fomc press conference", "fomc meeting minutes"],
    "ECB": ["main refinancing rate", "ecb press conference", "ecb monetary policy decision"]
}

def detailed_audit():
    df = pd.read_csv(AUDIT_FILE, low_memory=False)
    df["ts_raw"] = pd.to_datetime(df["timestamp_utc_raw"], errors="coerce")
    mask = (df["ts_raw"] >= "2015-01-01") & (df["ts_raw"] <= "2019-12-31")
    hist = df[mask & (df["impact_level"] == "HIGH")].copy()
    
    hist["event_lower"] = hist["raw_event_name"].str.lower().fillna("")
    
    summary = []
    for family, aliases in MAPPING.items():
        found = hist[hist["event_lower"].apply(lambda x: any(a in x for a in aliases))]
        count_by_year = found.groupby(hist["ts_raw"].dt.year).size()
        print(f"\n--- {family} (HIGH IMPACT) ---")
        print(count_by_year)
        summary.append(count_by_year)

if __name__ == "__main__":
    detailed_audit()
