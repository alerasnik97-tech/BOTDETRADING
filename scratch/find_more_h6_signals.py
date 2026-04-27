import pandas as pd
import sys
from pathlib import Path

# Add current dir to path
sys.path.append(str(Path.cwd()))

from scripts.h6_paper_shadow_runner import load_day_context, annotate_post_news_external_liquidity_shift_frame, filter_paper_signals
from research_lab.config import NY_TZ

def find_signals():
    data_dir = Path("data_precision/dukascopy")
    if not data_dir.exists():
        print(f"Directory {data_dir} not found")
        return

    # Scan for available dates (YYYY-MM-DD.csv)
    files = list(data_dir.glob("EURUSD_*.csv"))
    # The file names are EURUSD_mid.csv, EURUSD_bid.csv, etc. 
    # High precision package loads specific structure.
    
    # Let's try some known range
    dates_2024 = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    dates_2025 = pd.date_range("2025-01-01", "2025-12-31", freq="B")
    all_dates = list(dates_2024) + list(dates_2025)
    
    audited = ["2024-06-06", "2024-09-06", "2025-01-29", "2025-03-26", "2025-10-01"]
    
    found_dates = []
    print("Searching for signals in 10:00-11:00 NY window...")
    
    for dt in all_dates:
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in audited:
            continue
            
        try:
            m3_day, day_m1, m5_full, _ = load_day_context(date_str)
            # Use same logic as runner to keep consistency
            from research_lab.news_filter import require_operational_news
            from scripts.h6_paper_shadow_runner import paper_news_config
            
            news_result = require_operational_news("EURUSD", paper_news_config())
            annotated, raw_signals = annotate_post_news_external_liquidity_shift_frame(
                m3_day, m5_full, news_events=news_result.events, news_config=paper_news_config()
            )
            signals = filter_paper_signals(raw_signals)
            
            if not signals.empty:
                print(f"Found signal on {date_str}")
                found_dates.append(date_str)
                if len(found_dates) >= 10: # Collect a few more just in case
                    break
        except Exception:
            # Skip dates without data
            continue
            
    print("\nSummary of new signal dates found:")
    for d in found_dates:
        print(d)

if __name__ == "__main__":
    find_signals()
