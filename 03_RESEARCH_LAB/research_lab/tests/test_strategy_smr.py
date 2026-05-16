import unittest
import pandas as pd
from research_lab.strategies import strategy_smr

class TestStrategySMR(unittest.TestCase):
    def setUp(self):
        self.params = strategy_smr.default_params()
        # Mock data structure
        self.cols = [
            "open", "high", "low", "close", "rsi7", "atr14",
            "bb_mid_20_2_2", "bb_upper_20_2_2", "bb_lower_20_2_2"
        ]
        
    def create_mock_frame(self, data):
        return pd.DataFrame(data, columns=self.cols)

    def test_long_signal_valid(self):
        data = [
            # i-1: Close < Lower, RSI < 25
            {"open": 1.1000, "high": 1.1005, "low": 1.0990, "close": 1.0995, 
             "rsi7": 20.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1100, 
             "bb_upper_20_2_2": 1.1200, "bb_lower_20_2_2": 1.1000},
            # i: Open < Lower_i-1 (1.1000)
            {"open": 1.0996, "high": 1.1000, "low": 1.0990, "close": 1.0998, 
             "rsi7": 22.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1100, 
             "bb_upper_20_2_2": 1.1200, "bb_lower_20_2_2": 1.1000}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_smr.signal(frame, 1, self.params)
        
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "long")
        # SL check: 1.5 * ATR(14)
        self.assertEqual(sig["stop_atr"], 1.5)
        # TP check: Basis[i-1]
        self.assertEqual(sig["target_price"], 1.1100)

    def test_long_signal_cancel_reentry(self):
        data = [
            # i-1: Close < Lower, RSI < 25
            {"open": 1.1000, "high": 1.1005, "low": 1.0990, "close": 1.0995, 
             "rsi7": 20.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1100, 
             "bb_upper_20_2_2": 1.1200, "bb_lower_20_2_2": 1.1000},
            # i: Open >= Lower_i-1 (1.1000) -> CANCEL
            {"open": 1.1001, "high": 1.1005, "low": 1.1000, "close": 1.1003, 
             "rsi7": 30.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1100, 
             "bb_upper_20_2_2": 1.1200, "bb_lower_20_2_2": 1.1000}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_smr.signal(frame, 1, self.params)
        
        self.assertIsNone(sig)

    def test_short_signal_valid(self):
        data = [
            # i-1: Close > Upper, RSI > 75
            {"open": 1.1000, "high": 1.1105, "low": 1.1000, "close": 1.1101, 
             "rsi7": 80.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.0900, 
             "bb_upper_20_2_2": 1.1000, "bb_lower_20_2_2": 1.0800},
            # i: Open > Upper_i-1 (1.1000)
            {"open": 1.1102, "high": 1.1105, "low": 1.1100, "close": 1.1103, 
             "rsi7": 78.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.0900, 
             "bb_upper_20_2_2": 1.1000, "bb_lower_20_2_2": 1.0800}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_smr.signal(frame, 1, self.params)
        
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "short")
        self.assertEqual(sig["target_price"], 1.0900)

    def test_short_signal_cancel_reentry(self):
        data = [
            # i-1: Close > Upper, RSI > 75
            {"open": 1.1000, "high": 1.1105, "low": 1.1000, "close": 1.1101, 
             "rsi7": 80.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.0900, 
             "bb_upper_20_2_2": 1.1000, "bb_lower_20_2_2": 1.0800},
            # i: Open <= Upper_i-1 (1.1000) -> CANCEL
            {"open": 1.0999, "high": 1.1105, "low": 1.0990, "close": 1.0995, 
             "rsi7": 60.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.0900, 
             "bb_upper_20_2_2": 1.1000, "bb_lower_20_2_2": 1.0800}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_smr.signal(frame, 1, self.params)
        
        self.assertIsNone(sig)

    def test_filter_distance_too_small(self):
        data = [
            # i-1: Close < Lower, RSI < 25 BUT Dist < 5 pips (0.0005)
            {"open": 1.1000, "high": 1.1005, "low": 1.0990, "close": 1.0999, 
             "rsi7": 20.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1003, # Dist = 4 pips
             "bb_upper_20_2_2": 1.1100, "bb_lower_20_2_2": 1.1000},
            {"open": 1.0990, "high": 1.1000, "low": 1.0980, "close": 1.0990, 
             "rsi7": 15.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1003, 
             "bb_upper_20_2_2": 1.1100, "bb_lower_20_2_2": 1.1000}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_smr.signal(frame, 1, self.params)
        
        self.assertIsNone(sig)

    def test_filter_bandwidth_too_small(self):
        data = [
            # i-1: Close < Lower, RSI < 25 BUT Band Width < 10 pips (0.0010)
            {"open": 1.1000, "high": 1.1005, "low": 1.0990, "close": 1.0995, 
             "rsi7": 20.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1000, 
             "bb_upper_20_2_2": 1.1004, "bb_lower_20_2_2": 1.0996}, # Width = 8 pips
            {"open": 1.0990, "high": 1.1000, "low": 1.0980, "close": 1.0990, 
             "rsi7": 15.0, "atr14": 0.0020, "bb_mid_20_2_2": 1.1000, 
             "bb_upper_20_2_2": 1.1004, "bb_lower_20_2_2": 1.0996}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_smr.signal(frame, 1, self.params)
        
        self.assertIsNone(sig)

if __name__ == '__main__':
    unittest.main()
