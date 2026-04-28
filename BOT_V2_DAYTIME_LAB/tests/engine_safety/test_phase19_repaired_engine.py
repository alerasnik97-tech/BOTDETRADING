import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
sys.path.insert(0, str(ROOT / "BOT_V2_DAYTIME_LAB" / "src"))

from phase19_repaired_engine import (  # noqa: E402
    NativeM3UnavailableError,
    Phase19RepairedConfig,
    detect_first_m3_choch,
    get_confirmed_fractals,
    is_news_blocked,
    require_native_m3,
    simulate_repaired_backtest,
)


def m3_frame(rows):
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df


class TestPhase19RepairedEngine(unittest.TestCase):
    def test_no_native_m3_aborts(self):
        manifest = {"period_2020_2026": {"m5_bid": "x", "m5_ask": "y", "h1_bid": "z", "news": "n"}}
        with self.assertRaises(NativeM3UnavailableError):
            require_native_m3(manifest, "period_2020_2026")

    def test_entry_is_next_bar_open(self):
        cfg = Phase19RepairedConfig(fractal_n_m3=1)
        df = m3_frame(
            [
                {"timestamp": "2024-01-02T13:00:00Z", "open_bid": 1.1000, "high_bid": 1.1005, "low_bid": 1.0995, "close_bid": 1.1000, "open_ask": 1.1001, "high_ask": 1.1006, "low_ask": 1.0996, "close_ask": 1.1001},
                {"timestamp": "2024-01-02T13:03:00Z", "open_bid": 1.1000, "high_bid": 1.1010, "low_bid": 1.0998, "close_bid": 1.1005, "open_ask": 1.1001, "high_ask": 1.1011, "low_ask": 1.0999, "close_ask": 1.1006},
                {"timestamp": "2024-01-02T13:06:00Z", "open_bid": 1.1005, "high_bid": 1.1006, "low_bid": 1.0999, "close_bid": 1.1002, "open_ask": 1.1006, "high_ask": 1.1007, "low_ask": 1.1000, "close_ask": 1.1003},
                {"timestamp": "2024-01-02T13:09:00Z", "open_bid": 1.1002, "high_bid": 1.1020, "low_bid": 1.1000, "close_bid": 1.1015, "open_ask": 1.1003, "high_ask": 1.1021, "low_ask": 1.1001, "close_ask": 1.1016},
                {"timestamp": "2024-01-02T13:12:00Z", "open_bid": 1.1016, "high_bid": 1.1020, "low_bid": 1.1012, "close_bid": 1.1018, "open_ask": 1.1017, "high_ask": 1.1021, "low_ask": 1.1013, "close_ask": 1.1019},
            ]
        )
        sweeps = pd.DataFrame(
            [{"sweep_id": 1, "timestamp_ny": df.loc[2, "timestamp_ny"], "type": "BULLISH_SWEEP", "level_type": "h1_fractal_low", "peak_price": 1.0990}]
        )
        signals = detect_first_m3_choch(df, sweeps, cfg)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals.iloc[0]["choch_time"], df.loc[3, "timestamp_ny"])
        self.assertEqual(signals.iloc[0]["entry_time"], df.loc[4, "timestamp_ny"])
        self.assertAlmostEqual(signals.iloc[0]["entry_price"], df.loc[4, "open_ask"])

    def test_bid_ask_execution_long_and_short(self):
        cfg = Phase19RepairedConfig(max_trades_per_day=2, tp_r=1.0, start_time="08:00", end_time="20:00")
        df = m3_frame(
            [
                {"timestamp": "2024-01-02T13:00:00Z", "open_bid": 1.1000, "high_bid": 1.1001, "low_bid": 1.0999, "close_bid": 1.1000, "open_ask": 1.1001, "high_ask": 1.1002, "low_ask": 1.1000, "close_ask": 1.1001},
                {"timestamp": "2024-01-02T13:03:00Z", "open_bid": 1.1000, "high_bid": 1.1012, "low_bid": 1.0998, "close_bid": 1.1010, "open_ask": 1.1001, "high_ask": 1.1013, "low_ask": 1.0999, "close_ask": 1.1011},
                {"timestamp": "2024-01-02T13:06:00Z", "open_bid": 1.1010, "high_bid": 1.1022, "low_bid": 1.1008, "close_bid": 1.1020, "open_ask": 1.1011, "high_ask": 1.1023, "low_ask": 1.1009, "close_ask": 1.1021},
                {"timestamp": "2024-01-02T14:00:00Z", "open_bid": 1.2000, "high_bid": 1.2001, "low_bid": 1.1999, "close_bid": 1.2000, "open_ask": 1.2001, "high_ask": 1.2002, "low_ask": 1.2000, "close_ask": 1.2001},
                {"timestamp": "2024-01-02T14:03:00Z", "open_bid": 1.2000, "high_bid": 1.2002, "low_bid": 1.1988, "close_bid": 1.1990, "open_ask": 1.2001, "high_ask": 1.2003, "low_ask": 1.1989, "close_ask": 1.1991},
            ]
        )
        signals = pd.DataFrame(
            [
                {"signal_id": 1, "sweep_id": 1, "entry_time": df.loc[1, "timestamp_ny"], "entry_bar_index": 1, "direction": "LONG", "entry_price": 1.1001, "sl_price": 1.0990, "sweep_level_type": "a", "sweep_peak_price": 1.0990},
                {"signal_id": 2, "sweep_id": 2, "entry_time": df.loc[3, "timestamp_ny"], "entry_bar_index": 3, "direction": "SHORT", "entry_price": 1.2000, "sl_price": 1.2010, "sweep_level_type": "b", "sweep_peak_price": 1.2010},
            ]
        )
        trades = simulate_repaired_backtest(df, signals, pd.DataFrame(), cfg)
        self.assertEqual(trades.iloc[0]["status"], "TP")
        self.assertEqual(trades.iloc[1]["status"], "TP")
        self.assertAlmostEqual(trades.iloc[0]["entry_price"], 1.1001)
        self.assertAlmostEqual(trades.iloc[1]["entry_price"], 1.2000)

    def test_same_bar_conservative(self):
        cfg = Phase19RepairedConfig(tp_r=1.0, start_time="08:00", end_time="20:00")
        df = m3_frame(
            [
                {"timestamp": "2024-01-02T13:00:00Z", "open_bid": 1.1000, "high_bid": 1.1001, "low_bid": 1.0999, "close_bid": 1.1000, "open_ask": 1.1001, "high_ask": 1.1002, "low_ask": 1.1000, "close_ask": 1.1001},
                {"timestamp": "2024-01-02T13:03:00Z", "open_bid": 1.1000, "high_bid": 1.1030, "low_bid": 1.0980, "close_bid": 1.1010, "open_ask": 1.1001, "high_ask": 1.1031, "low_ask": 1.0981, "close_ask": 1.1011},
                {"timestamp": "2024-01-02T13:06:00Z", "open_bid": 1.1010, "high_bid": 1.1012, "low_bid": 1.0989, "close_bid": 1.0990, "open_ask": 1.1011, "high_ask": 1.1013, "low_ask": 1.0990, "close_ask": 1.0991},
            ]
        )
        signals = pd.DataFrame([{"signal_id": 1, "sweep_id": 1, "entry_time": df.loc[1, "timestamp_ny"], "entry_bar_index": 1, "direction": "LONG", "entry_price": 1.1001, "sl_price": 1.0990, "sweep_level_type": "a", "sweep_peak_price": 1.0990}])
        trades = simulate_repaired_backtest(df, signals, pd.DataFrame(), cfg)
        self.assertEqual(trades.iloc[0]["status"], "SL")
        self.assertFalse(bool(trades.iloc[0]["same_bar"]))

    def test_news_guard_blocks(self):
        cfg = Phase19RepairedConfig(news_guard_minutes=30)
        news = pd.DataFrame([{"timestamp": pd.Timestamp("2024-01-02T13:00:00Z"), "currency": "USD", "impact_level": "HIGH"}])
        self.assertTrue(is_news_blocked(pd.Timestamp("2024-01-02T13:10:00Z").tz_convert("America/New_York"), news, cfg))

    def test_forced_close_and_no_rollover(self):
        cfg = Phase19RepairedConfig(tp_r=3.0, start_time="19:00", end_time="20:00")
        df = m3_frame(
            [
                {"timestamp": "2024-01-03T00:54:00Z", "open_bid": 1.1000, "high_bid": 1.1001, "low_bid": 1.0999, "close_bid": 1.1000, "open_ask": 1.1001, "high_ask": 1.1002, "low_ask": 1.1000, "close_ask": 1.1001},
                {"timestamp": "2024-01-03T00:57:00Z", "open_bid": 1.1000, "high_bid": 1.1002, "low_bid": 1.0999, "close_bid": 1.1001, "open_ask": 1.1001, "high_ask": 1.1003, "low_ask": 1.1000, "close_ask": 1.1002},
                {"timestamp": "2024-01-03T01:00:00Z", "open_bid": 1.1001, "high_bid": 1.1002, "low_bid": 1.1000, "close_bid": 1.1001, "open_ask": 1.1002, "high_ask": 1.1003, "low_ask": 1.1001, "close_ask": 1.1002},
            ]
        )
        signals = pd.DataFrame([{"signal_id": 1, "sweep_id": 1, "entry_time": df.loc[0, "timestamp_ny"], "entry_bar_index": 0, "direction": "LONG", "entry_price": 1.1001, "sl_price": 1.0990, "sweep_level_type": "a", "sweep_peak_price": 1.0990}])
        trades = simulate_repaired_backtest(df, signals, pd.DataFrame(), cfg)
        self.assertEqual(trades.iloc[0]["status"], "FORCED_CLOSE_2000")
        self.assertEqual(trades.iloc[0]["exit_time"].strftime("%H:%M"), "20:00")
        self.assertEqual(trades.iloc[0]["entry_time"].date(), trades.iloc[0]["exit_time"].date())

    def test_no_overlap_and_max_trades_per_day(self):
        cfg = Phase19RepairedConfig(max_trades_per_day=1, tp_r=1.0, start_time="08:00", end_time="20:00")
        df = m3_frame(
            [
                {"timestamp": "2024-01-02T13:00:00Z", "open_bid": 1.1000, "high_bid": 1.1001, "low_bid": 1.0999, "close_bid": 1.1000, "open_ask": 1.1001, "high_ask": 1.1002, "low_ask": 1.1000, "close_ask": 1.1001},
                {"timestamp": "2024-01-02T13:03:00Z", "open_bid": 1.1000, "high_bid": 1.1002, "low_bid": 1.0998, "close_bid": 1.1000, "open_ask": 1.1001, "high_ask": 1.1003, "low_ask": 1.0999, "close_ask": 1.1001},
                {"timestamp": "2024-01-02T13:06:00Z", "open_bid": 1.1000, "high_bid": 1.1020, "low_bid": 1.0998, "close_bid": 1.1015, "open_ask": 1.1001, "high_ask": 1.1021, "low_ask": 1.0999, "close_ask": 1.1016},
                {"timestamp": "2024-01-02T13:09:00Z", "open_bid": 1.1015, "high_bid": 1.1020, "low_bid": 1.1010, "close_bid": 1.1015, "open_ask": 1.1016, "high_ask": 1.1021, "low_ask": 1.1011, "close_ask": 1.1016},
            ]
        )
        signals = pd.DataFrame(
            [
                {"signal_id": 1, "sweep_id": 1, "entry_time": df.loc[0, "timestamp_ny"], "entry_bar_index": 0, "direction": "LONG", "entry_price": 1.1001, "sl_price": 1.0990, "sweep_level_type": "a", "sweep_peak_price": 1.0990},
                {"signal_id": 2, "sweep_id": 2, "entry_time": df.loc[1, "timestamp_ny"], "entry_bar_index": 1, "direction": "LONG", "entry_price": 1.1001, "sl_price": 1.0990, "sweep_level_type": "b", "sweep_peak_price": 1.0990},
            ]
        )
        trades = simulate_repaired_backtest(df, signals, pd.DataFrame(), cfg)
        self.assertEqual(len(trades), 1)

    def test_h1_fractal_no_lookahead(self):
        df = pd.DataFrame({"high_bid": [1.10, 1.20, 1.10, 1.09], "low_bid": [1.00, 1.00, 1.00, 1.00]})
        fh, _ = get_confirmed_fractals(df, n=1)
        self.assertTrue(pd.isna(fh[1]))
        self.assertEqual(fh[2], 1.20)


if __name__ == "__main__":
    unittest.main()
