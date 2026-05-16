import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from research_lab.strategies import strategy_vse

class TestStrategyVSE(unittest.TestCase):
    def setUp(self):
        self.params = strategy_vse.default_params()
        self.params["min_squeeze_bars"] = 5
        self.params["limit_expiry_bars"] = 5
        self.cols = [
            "open", "high", "low", "close", "atr14",
            "bb_upper_20_2_0", "bb_lower_20_2_0", "kc_upper_20_1_5", "kc_lower_20_1_5"
        ]
        
    def create_vse_frame(self, events):
        # Create 100 bars total
        data = []
        base_time = datetime(2023, 1, 1, 10, 0)
        
        for i in range(100):
            # Default: Squeezed
            row = {
                "open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1000, "atr14": 0.0010,
                "bb_upper_20_2_0": 1.1005, "bb_lower_20_2_0": 1.0995, 
                "kc_upper_20_1_5": 1.1010, "kc_lower_20_1_5": 1.0990
            }
            # Overwrite with events
            if i in events:
                row.update(events[i])
            data.append(row)
            
        df = pd.DataFrame(data, columns=self.cols)
        df.index = [base_time + timedelta(minutes=5*i) for i in range(100)]
        return df

    def test_vse_valid_long_retest(self):
        # 10:00 + 40 bars = 13:20 (Window 13:00 - 16:00 OK)
        # Bar 40: Breakout Long (First Cross)
        # Bar 41: Outside
        # Bar 42: Retest Touch
        events = {
            40: {"open": 1.1000, "high": 1.1020, "low": 1.0990, "close": 1.1015, "bb_upper_20_2_0": 1.1009, "bb_lower_20_2_0": 1.1001},
            41: {"open": 1.1015, "high": 1.1015, "low": 1.1012, "close": 1.1014, "bb_upper_20_2_0": 1.1010, "bb_lower_20_2_0": 1.1000},
            42: {"open": 1.1014, "high": 1.1018, "low": 1.1008, "close": 1.1011, "bb_upper_20_2_0": 1.1011, "bb_lower_20_2_0": 1.1001}
        }
        # Squeeze check from 36 to 40 (5 bars)
        # j-min+1:j+1 -> 40-5+1:41 -> 36,37,38,39,40. All these MUST be squeezed.
        # My default rows are squeezed. Bar 40 is explicitly squeezed (1.1010 < 1.1020). OK.
        
        frame = self.create_vse_frame(events)
        sig = strategy_vse.signal(frame, 42, self.params)
        
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "long")
        self.assertAlmostEqual(sig["signal_price"], 1.1009)

    def test_vse_tp_cancellation_works(self):
        # Bar 40: Breakout Long. tp_price = 1.1010 + 0.0008 = 1.1018
        # Bar 41: High touches 1.1019 -> Cancel
        # Bar 42: Retest touch -> Should be None
        events = {
            40: {"open": 1.1000, "high": 1.1020, "low": 1.0990, "close": 1.1015, "bb_upper_20_2_0": 1.1010, "bb_lower_20_2_0": 1.1000},
            41: {"open": 1.1015, "high": 1.1019, "low": 1.1012, "close": 1.1017, "bb_upper_20_2_0": 1.1010, "bb_lower_20_2_0": 1.1000},
            42: {"open": 1.1017, "high": 1.1018, "low": 1.1009, "close": 1.1011, "bb_upper_20_2_0": 1.1011, "bb_lower_20_2_0": 1.1001}
        }
        frame = self.create_vse_frame(events)
        sig = strategy_vse.signal(frame, 42, self.params)
        self.assertIsNone(sig)

if __name__ == '__main__':
    unittest.main()
