import unittest
import pandas as pd
import numpy as np
from research_lab.strategies import strategy_ls_sr

class TestStrategyLSSR(unittest.TestCase):
    def setUp(self):
        self.params = strategy_ls_sr.default_params()
        self.cols = [
            "open", "high", "low", "close",
            "session_range_high_11_00", "session_range_low_11_00", "session_range_complete_11_00"
        ]
        
    def create_mock_frame(self, data):
        return pd.DataFrame(data, columns=self.cols)

    def test_am_range_blocking(self):
        # Rango AM: 1.1020 - 1.1010 = 10 pips (< 15 pips)
        data = [
            {"open": 1.1015, "high": 1.1016, "low": 1.1009, "close": 1.1012, 
             "session_range_high_11_00": 1.1020, "session_range_low_11_00": 1.1010, "session_range_complete_11_00": True}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_ls_sr.signal(frame, 0, self.params)
        self.assertIsNone(sig)

    def test_short_sweep_valid(self):
        # Rango AM: 1.1050 - 1.1000 = 50 pips (> 15)
        # Sweep: High 1.1051 (> 1.1050 + 0.5 pips -> 1.10505)
        # Rechazo: Close 1.1045 (< 1.1050)
        # Intencion: (1.1051 - 1.1045) / (1.1051 - 1.1040) = 0.0006 / 0.0011 = 0.54 (> 0.5)
        data = [
            {"open": 1.1042, "high": 1.1051, "low": 1.1040, "close": 1.1045, 
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_ls_sr.signal(frame, 0, self.params)
        
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "short")
        # SL: High + 1 pip = 1.1051 + 0.0001 = 1.1052
        self.assertAlmostEqual(sig["stop_price"], 1.1052)
        # TP: 50% range = 1.1000 + 25 pips = 1.1025
        self.assertAlmostEqual(sig["target_price"], 1.1025)

    def test_short_sweep_invalid_intention(self):
        # Sweep y Rechazo OK, pero Intencion FAIL (Cierre en mitad superior)
        # Intencion: (1.1051 - 1.1048) / (1.1051 - 1.1040) = 0.0003 / 0.0011 = 0.27 (< 0.5)
        data = [
            {"open": 1.1042, "high": 1.1051, "low": 1.1040, "close": 1.1048, 
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_ls_sr.signal(frame, 0, self.params)
        self.assertIsNone(sig)

    def test_long_sweep_valid(self):
        # Rango AM: 1.1050 - 1.1000
        # Sweep: Low 1.0999 (< 1.1000 - 0.5 pips -> 1.09995)
        # Rechazo: Close 1.1005 (> 1.1000)
        # Intencion: (1.1005 - 1.0999) / (1.1010 - 1.0999) = 0.0006 / 0.0011 = 0.54 (> 0.5)
        data = [
            {"open": 1.1008, "high": 1.1010, "low": 1.0999, "close": 1.1005, 
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_ls_sr.signal(frame, 0, self.params)
        
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "long")
        # SL: Low - 1 pip = 1.0999 - 0.0001 = 1.0998
        self.assertAlmostEqual(sig["stop_price"], 1.0998)
        # TP: 1.1025
        self.assertAlmostEqual(sig["target_price"], 1.1025)

    def test_flat_bar_rejection(self):
        data = [
            {"open": 1.1051, "high": 1.1051, "low": 1.1051, "close": 1.1051, 
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_ls_sr.signal(frame, 0, self.params)
        self.assertIsNone(sig)

if __name__ == '__main__':
    unittest.main()
