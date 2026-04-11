from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig, NY_TZ
from research_lab.engine import entry_open_index, run_backtest


def make_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"], tz=NY_TZ) for item in rows])
    frame = pd.DataFrame(
        {
            "open": [item["open"] for item in rows],
            "high": [item["high"] for item in rows],
            "low": [item["low"] for item in rows],
            "close": [item["close"] for item in rows],
            "atr14": [item.get("atr14", 0.0010) for item in rows],
            "range_atr": [item.get("range_atr", 0.5) for item in rows],
        },
        index=index,
    )
    return frame


class LongSignalStrategy:
    NAME = "long_signal"
    WARMUP_BARS = 0

    @staticmethod
    def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
        signal_times = set(params.get("signal_times", []))
        if frame.index[i].strftime("%H:%M") in signal_times:
            return {
                "direction": params.get("direction", "long"),
                "stop_mode": "atr",
                "stop_atr": params.get("stop_atr", 1.0),
                "target_rr": params.get("target_rr", 1.0),
                "break_even_at_r": params.get("break_even_at_r"),
                "trailing_atr": False,
                "session_name": "light_fixed",
            }
        return None


class EngineTests(unittest.TestCase):
    def test_signal_before_11_does_not_open_trade(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:30:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1001},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1001, "high": 1.1004, "low": 1.0999, "close": 1.1003},
            ]
        )
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["10:45"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0, "session_name": "light_fixed"},
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.0, max_spread_pips=2.0, slippage_pips=0.0, commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 0)

    def test_entry_time_uses_next_bar_open_time(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1004, "low": 1.0998, "close": 1.1002},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1002, "high": 1.1005, "low": 1.1000, "close": 1.1004},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1004, "high": 1.1018, "low": 1.1003, "close": 1.1015},
                {"timestamp": "2022-01-03 11:30:00", "open": 1.1015, "high": 1.1016, "low": 1.1010, "close": 1.1012},
            ]
        )
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.0, max_spread_pips=2.0, slippage_pips=0.0, commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 1)
        entry_time = pd.to_datetime(result.trades.iloc[0]["entry_time"], utc=True).tz_convert(NY_TZ)
        self.assertEqual(entry_time.strftime("%H:%M"), "11:00")

    def test_boundary_fill_on_19_label_is_recorded_at_1845_open(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 18:30:00", "open": 1.1000, "high": 1.1005, "low": 1.0998, "close": 1.1002},
                {"timestamp": "2022-01-03 18:45:00", "open": 1.1002, "high": 1.1007, "low": 1.1000, "close": 1.1005},
                {"timestamp": "2022-01-03 19:00:00", "open": 1.1005, "high": 1.1009, "low": 1.1001, "close": 1.1006},
            ]
        )
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["18:45"], "direction": "long", "stop_atr": 1.0, "target_rr": 10.0, "session_name": "light_fixed"},
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.0, max_spread_pips=2.0, slippage_pips=0.0, commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 1)
        entry_time = pd.to_datetime(result.trades.iloc[0]["entry_time"], utc=True).tz_convert(NY_TZ)
        self.assertEqual(entry_time.strftime("%H:%M"), "18:45")
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "forced_session_close")

    def test_intrabar_ambiguous_bar_uses_stop_first_policy(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1001, "atr14": 0.0005},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1000, "high": 1.1012, "low": 1.0988, "close": 1.1005, "atr14": 0.0005},
            ]
        )
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0, "session_name": "light_fixed"},
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=0.0, max_spread_pips=2.0, slippage_pips=0.0, commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2, intrabar_exit_priority="stop_first"),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "stop_loss")

    def test_short_target_exit_includes_spread_and_slippage(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1000, "high": 1.1001, "low": 1.0997, "close": 1.0999},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1000, "high": 1.1002, "low": 1.0984, "close": 1.0986},
                {"timestamp": "2022-01-03 11:30:00", "open": 1.0986, "high": 1.0989, "low": 1.0982, "close": 1.0985},
            ]
        )
        config = EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.0, max_spread_pips=2.0, slippage_pips=0.5, commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2)
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "short", "stop_atr": 1.0, "target_rr": 1.0},
            config,
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertAlmostEqual(float(trade["entry_price"]), 1.09995, places=5)
        self.assertAlmostEqual(float(trade["exit_price"]), 1.0987375, places=5)

    def test_trade_count_is_unchanged_by_costs_when_spread_guard_does_not_block(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1000, "high": 1.1003, "low": 1.0997, "close": 1.1001},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1000, "high": 1.1015, "low": 1.0990, "close": 1.1010},
                {"timestamp": "2022-01-03 11:30:00", "open": 1.1010, "high": 1.1012, "low": 1.1004, "close": 1.1008},
            ]
        )
        params = {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0}
        no_cost = run_backtest(
            LongSignalStrategy,
            frame,
            params,
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=0.0, max_spread_pips=10.0, slippage_pips=0.0, commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        with_cost = run_backtest(
            LongSignalStrategy,
            frame,
            params,
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.2, max_spread_pips=10.0, slippage_pips=0.2, commission_per_lot_roundturn_usd=7.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(no_cost.trades), 1)
        self.assertEqual(len(with_cost.trades), 1)

    def test_final_close_applies_exit_costs(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1001},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1001, "high": 1.1003, "low": 1.1000, "close": 1.1002},
            ]
        )
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 10.0, "target_rr": 10.0, "session_name": "all_day"},
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.2, max_spread_pips=10.0, slippage_pips=0.2, commission_per_lot_roundturn_usd=7.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "final_bar_close")
        self.assertLess(float(result.trades.iloc[0]["pnl_usd"]), 0.0)

    def test_entry_open_index_respects_dst_offsets(self) -> None:
        close_index_dst_start = pd.DatetimeIndex(
            [
                pd.Timestamp("2022-03-14 11:00:00", tz=NY_TZ),
                pd.Timestamp("2022-03-14 11:15:00", tz=NY_TZ),
            ]
        )
        close_index_dst_end = pd.DatetimeIndex(
            [
                pd.Timestamp("2022-11-07 11:00:00", tz=NY_TZ),
                pd.Timestamp("2022-11-07 11:15:00", tz=NY_TZ),
            ]
        )
        open_index_start = entry_open_index(close_index_dst_start)
        open_index_end = entry_open_index(close_index_dst_end)
        self.assertEqual(open_index_start[1].strftime("%Y-%m-%d %H:%M %z"), "2022-03-14 11:00 -0400")
        self.assertEqual(open_index_end[1].strftime("%Y-%m-%d %H:%M %z"), "2022-11-07 11:00 -0500")
