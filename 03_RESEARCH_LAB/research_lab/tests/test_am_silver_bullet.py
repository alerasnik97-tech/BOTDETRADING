import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from research_lab.strategies import am_silver_bullet_ny
from research_lab.config import NY_TZ

class TestAmSilverBullet(unittest.TestCase):
    def setUp(self):
        # Create a mock M5 frame
        times = pd.date_range("2024-01-02 00:00", "2024-01-02 23:55", freq="5min", tz=NY_TZ)
        self.frame = pd.DataFrame(
            index=times,
            data={
                "open": 1.1000,
                "high": 1.1005,
                "low": 1.0995,
                "close": 1.1000,
                "volume": 1000,
                "atr14": 0.0005,
                "day_running_high": 1.1005,
                "day_running_low": 1.0995,
                "session_range_high_03_00_08_30": 1.1010,
                "session_range_low_03_00_08_30": 1.0990,
                "bullish_choch": False,
                "bearish_choch": False,
                "bullish_fvg": False,
                "bearish_fvg": False,
                "bullish_fvg_mid": np.nan,
                "bearish_fvg_mid": np.nan,
            }
        )
        self.params = am_silver_bullet_ny.default_params()

    def test_sweep_logic_long(self):
        # Setup conditions for a Long SB trade at 10:15
        idx_1015 = self.frame.index.get_loc(self.frame.at_time("10:15").index[0])
        
        # 1. Sweep happens (Low below anchor_low)
        self.frame.loc[self.frame.index[:idx_1015+1], "day_running_low"] = 1.0985 # Anchor was 1.0990
        
        # 2. Bull MSS (CHoCH) inside the setup lookback
        self.frame.at[self.frame.index[idx_1015-1], "bullish_choch"] = True
        
        # 3. FVG nearby (at i-1)
        self.frame.at[self.frame.index[idx_1015-1], "bullish_fvg"] = True
        self.frame.at[self.frame.index[idx_1015-1], "bullish_fvg_mid"] = 1.0995
        
        sig = am_silver_bullet_ny.signal(self.frame, idx_1015, self.params)
        
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "long")
        self.assertEqual(sig["entry_mode"], "limit")
        self.assertEqual(sig["limit_price"], 1.0995)

    def test_news_fortress_window_gate(self):
        # Testing outside 10:00-11:00 window
        idx_0945 = self.frame.index.get_loc(self.frame.at_time("09:45").index[0])
        self.frame.at[self.frame.index[idx_0945], "bullish_choch"] = True
        self.frame.at[self.frame.index[idx_0945-1], "bullish_fvg"] = True
        
        sig = am_silver_bullet_ny.signal(self.frame, idx_0945, self.params)
        self.assertIsNone(sig, "Trade detected outside the 10:00-11:00 window")

if __name__ == "__main__":
    unittest.main()
