
import unittest
import pandas as pd
import sys
import os

sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")
from v6_utils.causal import CausalClock, CausalDataFrame, LookAheadError

class TestCausal(unittest.TestCase):
    def test_clock_only_forward(self):
        clock = CausalClock(pd.Timestamp("2024-01-01", tz="UTC"))
        clock.advance_to(pd.Timestamp("2024-01-02", tz="UTC"))
        with self.assertRaises(ValueError):
            clock.advance_to(pd.Timestamp("2024-01-01", tz="UTC"))

    def test_lookahead_error_raised(self):
        df = pd.DataFrame({"val": [1, 2, 3]}, 
                          index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True))
        clock = CausalClock(pd.Timestamp("2024-01-02", tz="UTC"))
        cdf = CausalDataFrame(df, clock)
        
        # Acceso permitido (usando slice para disparar __getitem__)
        self.assertEqual(len(cdf.loc[:]), 2)
        
        # Acceso prohibido (loc directo al futuro)
        with self.assertRaises(LookAheadError):
            _ = cdf.loc["2024-01-03"]

    def test_causal_log_singleton(self):
        from v6_utils.causal import CausalLog
        log1 = CausalLog()
        log2 = CausalLog()
        self.assertIs(log1, log2)

    def test_causal_df_loc_filter(self):
        df = pd.DataFrame({"val": [10, 20, 30]}, 
                          index=pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:10"], utc=True))
        clock = CausalClock(pd.Timestamp("2024-01-01 10:06", tz="UTC"))
        cdf = CausalDataFrame(df, clock)
        self.assertEqual(len(cdf.loc[:]), 2)
        self.assertEqual(cdf.loc[:].index.max(), pd.Timestamp("2024-01-01 10:05", tz="UTC"))

if __name__ == '__main__':
    unittest.main()
