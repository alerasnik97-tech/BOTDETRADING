
import unittest
import pandas as pd
import sys
import os

sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")
from v6_utils.execution import next_bar_execute, next_bar_execute_limit, simulate_exit, simulate_exit_with_be, NoFillError

class TestExecution(unittest.TestCase):
    # ... (existing tests)

    def test_be_armed_then_tp(self):
        # Long entry at 1.1000
        # Ticks touch 1.1004 (trigger), then 1.1014 (TP)
        ticks = pd.DataFrame({"bid": [1.1002, 1.1004, 1.1015], "ask": [1.1004, 1.1006, 1.1017]}, 
                            index=pd.date_range("2024-01-01 10:00:01", periods=3, freq="1s", tz="UTC"))
        res = simulate_exit_with_be("long", 1.1000, 1.0990, 1.1004, 1.1000, 1.1014, ticks, ticks.index[0] - pd.Timedelta("1ms"))
        self.assertEqual(res.reason, "TP")
        self.assertEqual(res.fill_price, 1.1014)

    def test_be_armed_then_be_sl(self):
        # Long entry at 1.1000
        # Ticks touch 1.1004 (trigger), then drop to 1.1000 (new SL)
        ticks = pd.DataFrame({"bid": [1.1002, 1.1004, 1.0999], "ask": [1.1004, 1.1006, 1.1001]}, 
                            index=pd.date_range("2024-01-01 10:00:01", periods=3, freq="1s", tz="UTC"))
        res = simulate_exit_with_be("long", 1.1000, 1.0990, 1.1004, 1.1000, 1.1014, ticks, ticks.index[0] - pd.Timedelta("1ms"))
        self.assertEqual(res.reason, "BE-SL")
        self.assertEqual(res.fill_price, 1.1000)

    def test_be_not_armed_sl(self):
        # Long entry at 1.1000
        # Ticks drop to 1.0990 WITHOUT touching 1.1004
        ticks = pd.DataFrame({"bid": [1.1002, 1.1003, 1.0989], "ask": [1.1004, 1.1005, 1.0991]}, 
                            index=pd.date_range("2024-01-01 10:00:01", periods=3, freq="1s", tz="UTC"))
        res = simulate_exit_with_be("long", 1.1000, 1.0990, 1.1004, 1.1000, 1.1014, ticks, ticks.index[0] - pd.Timedelta("1ms"))
        self.assertEqual(res.reason, "SL")
        self.assertEqual(res.fill_price, 1.0990)

    def test_be_armed_at_exact_threshold(self):
        # First tick post-fill exactly at be_trigger
        ticks = pd.DataFrame({"bid": [1.1004, 1.0999], "ask": [1.1006, 1.1001]}, 
                            index=pd.date_range("2024-01-01 10:00:01", periods=2, freq="1s", tz="UTC"))
        res = simulate_exit_with_be("long", 1.1000, 1.0990, 1.1004, 1.1000, 1.1014, ticks, ticks.index[0] - pd.Timedelta("1ms"))
        self.assertEqual(res.reason, "BE-SL")
        self.assertEqual(res.fill_price, 1.1000)
    def test_nbe_long_at_ask(self):
        # Signal close at 10:00:00
        signal_close = pd.Timestamp("2024-03-10 10:00:00", tz="UTC")
        ticks = pd.DataFrame({
            "bid": [1.1000, 1.1010],
            "ask": [1.1002, 1.1012]
        }, index=pd.to_datetime(["2024-03-10 10:00:00.100", "2024-03-10 10:00:00.500"], utc=True))
        
        res = next_bar_execute("long", signal_close, ticks)
        self.assertEqual(res.fill_price, 1.1002)
        self.assertEqual(res.fill_time, ticks.index[0])

    def test_simulate_exit_sl_first_intra_bar(self):
        entry_price = 1.1000
        sl = 1.0990
        tp = 1.1020
        # Ticks: baja a SL primero, luego sube a TP
        ticks = pd.DataFrame({
            "bid": [1.0995, 1.0989, 1.1025],
            "ask": [1.0997, 1.0991, 1.1027]
        }, index=pd.date_range("2024-03-10 10:00:01", periods=3, freq="1s", tz="UTC"))
        
        # Fill time es el primer tick (10:00:01)
        fill_time = ticks.index[0]
        res = simulate_exit("long", entry_price, sl, tp, ticks, fill_time=fill_time)
        self.assertEqual(res.reason, "SL")
        self.assertEqual(res.fill_price, sl)
        self.assertEqual(res.fill_time, ticks.index[1])

    def test_nbe_short_at_bid(self):
        signal_close = pd.Timestamp("2024-03-10 10:00:00", tz="UTC")
        ticks = pd.DataFrame({"bid": [1.1000], "ask": [1.1002]}, 
                            index=pd.to_datetime(["2024-03-10 10:00:00.100"], utc=True))
        res = next_bar_execute("short", signal_close, ticks)
        self.assertEqual(res.fill_price, 1.1000)

    def test_no_fill_on_weekend_gap(self):
        signal_close = pd.Timestamp("2024-03-08 16:59:00", tz="UTC") # Friday close
        # Asegurar que el df vacío tenga DatetimeIndex
        empty_ticks = pd.DataFrame(columns=["bid", "ask"], index=pd.to_datetime([], utc=True))
        with self.assertRaises(NoFillError):
            next_bar_execute("long", signal_close, empty_ticks)

    def test_simulate_exit_tp_first(self):
        ticks = pd.DataFrame({"bid": [1.1005, 1.1015], "ask": [1.1007, 1.1017]}, 
                             index=pd.date_range("2024-01-01 10:00:00", periods=2, freq="1s", tz="UTC"))
        # Fill time es justo antes del primer tick útil
        fill_time = pd.Timestamp("2024-01-01 09:59:59", tz="UTC")
        res = simulate_exit("long", 1.1000, 1.0900, 1.1010, ticks, fill_time=fill_time)
        self.assertEqual(res.reason, "TP")

    def test_time_exit_long(self):
        t_exit = pd.Timestamp("2024-01-01 10:00:05", tz="UTC")
        ticks = pd.DataFrame({"bid": [1.1000, 1.1005], "ask": [1.1002, 1.1007]}, 
                            index=pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:00:10"], utc=True))
        # Fill time es el primer tick
        fill_time = ticks.index[0]
        res = simulate_exit("long", 1.1000, 1.0900, 1.1100, ticks, fill_time=fill_time, time_exit=t_exit)
        self.assertEqual(res.reason, "TIME")

    def test_limit_long_fills_at_limit(self):
        signal_t = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
        ticks = pd.DataFrame({
            "bid": [1.1005, 1.0998, 1.0996],
            "ask": [1.1007, 1.1000, 1.0998]
        }, index=pd.to_datetime(["2024-01-01 10:05:00", "2024-01-01 10:10:00", "2024-01-01 10:15:00"], utc=True))
        res = next_bar_execute_limit("long", signal_t, 1.10000, ticks, 30)
        self.assertIsNotNone(res)
        self.assertEqual(res.fill_time, pd.Timestamp("2024-01-01 10:10:00", tz="UTC"))
        self.assertEqual(res.fill_price, 1.10000)

    def test_limit_short_fills_at_limit(self):
        signal_t = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
        ticks = pd.DataFrame({
            "bid": [1.0995, 1.1000, 1.1002],
            "ask": [1.0997, 1.1002, 1.1004]
        }, index=pd.to_datetime(["2024-01-01 10:05:00", "2024-01-01 10:10:00", "2024-01-01 10:15:00"], utc=True))
        res = next_bar_execute_limit("short", signal_t, 1.10000, ticks, 30)
        self.assertIsNotNone(res)
        self.assertEqual(res.fill_time, pd.Timestamp("2024-01-01 10:10:00", tz="UTC"))
        self.assertEqual(res.fill_price, 1.10000)

    def test_limit_tif_expires(self):
        signal_t = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
        ticks = pd.DataFrame({
            "bid": [1.1005],
            "ask": [1.1007]
        }, index=pd.to_datetime(["2024-01-01 10:05:00"], utc=True))
        # Expiration at 10:30. Tick at 10:05 is not enough. 
        # If there are no more ticks, it should return None.
        res = next_bar_execute_limit("long", signal_t, 1.1000, ticks, 30)
        self.assertIsNone(res)

    def test_limit_uses_ask_not_bid_for_long(self):
        signal_t = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
        ticks = pd.DataFrame({
            "bid": [1.0999], # Bid touches limit
            "ask": [1.1001]  # Ask does NOT touch limit
        }, index=pd.to_datetime(["2024-01-01 10:05:00"], utc=True))
        res = next_bar_execute_limit("long", signal_t, 1.1000, ticks, 30)
        self.assertIsNone(res)

    def test_limit_does_not_fill_at_signal_tick(self):
        signal_t = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
        ticks = pd.DataFrame({
            "bid": [1.0998],
            "ask": [1.1000]
        }, index=pd.to_datetime(["2024-01-01 10:00:00"], utc=True)) # Tick exactly at signal_time
        res = next_bar_execute_limit("long", signal_t, 1.1000, ticks, 30)
        self.assertIsNone(res)

if __name__ == '__main__':
    unittest.main()
