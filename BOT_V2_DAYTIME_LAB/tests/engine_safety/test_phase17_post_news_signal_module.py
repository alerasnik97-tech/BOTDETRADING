
import pandas as pd
import numpy as np
from datetime import datetime, time
from phase17_post_news_signal_module import PostNewsSignalModule

def test_signal_module():
    print("Starting Synthetic Tests for PostNewsSignalModule...")
    module = PostNewsSignalModule()
    
    # Create synthetic price data (M5)
    times = pd.date_range("2025-01-01 07:00:00", periods=100, freq="5min")
    df_prices = pd.DataFrame({
        "timestamp_ny": times,
        "open_bid": 1.1000, "high_bid": 1.1005, "low_bid": 1.0995, "close_bid": 1.1000
    })
    
    # Scenario 1: CPI at 08:30. Block 60m (09:30). Signal at 09:40.
    # We simulate a range during block [08:30, 09:30]
    # Bar at 08:30 to 09:30 has range [1.0990, 1.1010]
    df_prices.loc[df_prices['timestamp_ny'] == '2025-01-01 08:30:00', 'high_bid'] = 1.1010
    df_prices.loc[df_prices['timestamp_ny'] == '2025-01-01 08:30:00', 'low_bid'] = 1.0990
    
    # Bar at 09:40 closes above 1.1010
    df_prices.loc[df_prices['timestamp_ny'] == '2025-01-01 09:40:00', 'close_bid'] = 1.1015
    
    df_news = pd.DataFrame([{
        "timestamp_ny": "2025-01-01 08:30:00",
        "event_name_normalized": "Core CPI m/m",
        "impact_level": "HIGH",
        "currency": "USD"
    }])
    
    sigs = module.generate_signals(df_prices, df_news)
    
    signal_bar = sigs[sigs['signal'] != 0]
    assert len(signal_bar) == 1, f"Should have 1 signal, found {len(signal_bar)}"
    assert signal_bar.iloc[0]['timestamp_ny'].time() == time(9, 40), "Signal should be at 09:40"
    assert signal_bar.iloc[0]['event_family'] == "CPI", "Family should be CPI"
    
    # Scenario 2: FOMC (Rejected)
    df_news_fomc = pd.DataFrame([{
        "timestamp_ny": "2025-01-01 14:00:00",
        "event_name_normalized": "FOMC Statement",
        "impact_level": "HIGH",
        "currency": "USD"
    }])
    sigs_fomc = module.generate_signals(df_prices, df_news_fomc)
    assert len(sigs_fomc[sigs_fomc['signal'] != 0]) == 0, "FOMC should be blocked"
    
    print("All Synthetic Tests PASSED.")

if __name__ == "__main__":
    test_signal_module()
