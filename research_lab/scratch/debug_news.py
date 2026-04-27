from pathlib import Path
import pandas as pd
import numpy as np
from research_lab.config import DEFAULT_PAIR, DEFAULT_DATA_DIRS, NewsConfig, NY_TZ
from research_lab.news_filter import load_news_events, build_entry_block

def debug_news():
    settings = NewsConfig(enabled=True) # Usamos los valores por defecto
    res = load_news_events(DEFAULT_PAIR, settings)
    print(f"News Enabled: {res.enabled}")
    if not res.enabled:
        print(f"Disabled Reason: {res.disabled_reason}")
    
    events = res.events
    print(f"Total Events: {len(events)}")
    
    if not events.empty:
        events["ts"] = pd.to_datetime(events["timestamp_ny"], utc=True).dt.tz_convert(NY_TZ)
        events_2024 = events[(events["ts"] >= "2024-01-01") & (events["ts"] <= "2025-01-01")]
        print(f"Events in 2024: {len(events_2024)}")
        if not events_2024.empty:
            print("First 5 events in 2024:")
            print(events_2024[["event_name_normalized", "timestamp_ny", "impact_level"]].head())

if __name__ == "__main__":
    debug_news()
