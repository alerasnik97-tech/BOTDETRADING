import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from research_lab.strategies import strategy_src

class TestStrategySRC(unittest.TestCase):
    def setUp(self):
        self.params = strategy_src.default_params()
        self.cols = [
            "open", "high", "low", "close", "atr14",
            "session_range_high_11_00", "session_range_low_11_00", "session_range_complete_11_00"
        ]
        # Timestamps para las 11:00+ NY
        from datetime import timedelta
        base = datetime(2023, 1, 1, 11, 0)
        self.times = [base + timedelta(minutes=5*i) for i in range(50)]
        
    def create_mock_frame(self, data):
        df = pd.DataFrame(data, columns=self.cols)
        df.index = self.times[:len(data)]
        return df

    def test_long_src_valid(self):
        # Max_AM: 1.1050, Min_AM: 1.1000 (Range 50 pips > 15)
        # 1. Ruptura (Bar 0): Close > 1.10505
        # 2. Outside (Bar 1): Low > 1.10505
        # 3. Retest + Conf (Bar 2): Touches 1.1050, Close > Open
        data = [
            {"open": 1.1050, "high": 1.1060, "low": 1.1045, "close": 1.1055, "atr14": 0.0010,
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True},
            {"open": 1.1055, "high": 1.1070, "low": 1.1060, "close": 1.1065, "atr14": 0.0010,
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True},
            {"open": 1.1060, "high": 1.1065, "low": 1.1050, "close": 1.1052, "atr14": 0.0010, # Touch zone [1.1049, 1.1051] NO! zona es +- 1 pip -> [1.1049, 1.1051]
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}
        ]
        # Ajuste: Retest zone es Max_AM +- 1 pip = [1.1049, 1.1051]
        # Gatillo: Close > Open. En Bar 2: Open 1.1060, Close 1.1052 -> Vela Roja -> No entry
        # Hagamos Bar 3 la entrada
        data.append({"open": 1.1050, "high": 1.1055, "low": 1.10495, "close": 1.1053, "atr14": 0.0010,
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True})
        
        frame = self.create_mock_frame(data)
        # Evaluamos en i=2
        sig_2 = strategy_src.signal(frame, 2, self.params)
        self.assertIsNone(sig_2) # Porque Close < Open
        
        # Evaluamos en i=3
        sig_3 = strategy_src.signal(frame, 3, self.params)
        self.assertIsNotNone(sig_3)
        self.assertEqual(sig_3["direction"], "long")
        # SL: Low retest bar (1.10495) - 1 pip (0.0001) = 1.10485
        self.assertAlmostEqual(sig_3["stop_price"], 1.10485)

    def test_long_src_must_come_from_outside(self):
        # Ruptura en Bar 0, pero Bar 1 NO es "totalmente afuera" (low <= 1.10505)
        # Bar 2 toca zona pero no hubo "outside" previo
        data = [
            {"open": 1.1050, "high": 1.1060, "low": 1.1045, "close": 1.1055, "atr14": 0.0010,
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True},
            {"open": 1.1055, "high": 1.1060, "low": 1.1050, "close": 1.1058, "atr14": 0.0010, # Low 1.1050 <= Max_AM + Buffer
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True},
            {"open": 1.1058, "high": 1.1060, "low": 1.1050, "close": 1.1055, "atr14": 0.0010,
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_src.signal(frame, 2, self.params)
        self.assertIsNone(sig)

    def test_src_invalidation_midpoint(self):
        # Ruptura OK, Outside OK, pero antes del retest hay un Close < Midpoint (1.1025)
        data = [
            {"open": 1.1050, "high": 1.1060, "low": 1.1045, "close": 1.1055, "atr14": 0.0010,
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True},
            {"open": 1.1055, "high": 1.1070, "low": 1.1060, "close": 1.1065, "atr14": 0.0010,
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True},
            {"open": 1.1065, "high": 1.1065, "low": 1.1020, "close": 1.1022, "atr14": 0.0010, # Close < 1.1025
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True},
            {"open": 1.1022, "high": 1.1055, "low": 1.1020, "close": 1.1052, "atr14": 0.0010, # Retest + Conf
             "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}
        ]
        frame = self.create_mock_frame(data)
        sig = strategy_src.signal(frame, 3, self.params)
        self.assertIsNone(sig)

    def test_src_expiry_18_bars(self):
        # Ruptura en i=0. Retest ocurre en i=20 (Excede 18 bars)
        data = [{"open": 1.1055, "close": 1.1055, "high": 1.1070, "low": 1.1060, "atr14": 0.0010,
                 "session_range_high_11_00": 1.1050, "session_range_low_11_00": 1.1000, "session_range_complete_11_00": True}] * 30
        df = pd.DataFrame(data)
        from datetime import timedelta
        base = datetime(2023, 1, 1, 11, 0)
        df.index = [base + timedelta(minutes=5*i) for i in range(30)]
        
        # Forzar breakout en i=0
        df.iloc[0, df.columns.get_loc("close")] = 1.1056
        # Forzar outside en i=1
        df.iloc[1, df.columns.get_loc("low")] = 1.1051
        # Forzar retest en i=20
        df.iloc[20, df.columns.get_loc("low")] = 1.1050
        df.iloc[20, df.columns.get_loc("open")] = 1.1049
        df.iloc[20, df.columns.get_loc("close")] = 1.1051 # Conf OK
        
        sig = strategy_src.signal(df, 20, self.params)
        self.assertIsNone(sig)

if __name__ == '__main__':
    unittest.main()
