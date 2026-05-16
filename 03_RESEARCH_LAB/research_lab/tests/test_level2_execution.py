from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig, NY_TZ
from research_lab.engine import run_backtest
from research_lab.report import build_trades_export


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
    NAME = "long_signal_level2"
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
                "session_name": params.get("session_name", "light_fixed"),
            }
        return None


class Level2ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.news_block = np.zeros(4, dtype=bool)

    def _base_config(self, **overrides) -> EngineConfig:
        payload = {
            "pair": "EURUSD",
            "risk_pct": 0.5,
            "assumed_spread_pips": 1.0,
            "max_spread_pips": 10.0,
            "slippage_pips": 0.5,
            "commission_per_lot_roundturn_usd": 7.0,
            "max_trades_per_day": 2,
        }
        payload.update(overrides)
        return EngineConfig(**payload)

    def _target_frame(self) -> pd.DataFrame:
        return make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1001},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1000, "high": 1.1015, "low": 1.0995, "close": 1.1010},
                {"timestamp": "2022-01-03 11:30:00", "open": 1.1010, "high": 1.1012, "low": 1.1006, "close": 1.1009},
            ]
        )

    def _forced_close_frame(self) -> pd.DataFrame:
        return make_frame(
            [
                {"timestamp": "2022-01-03 18:30:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 18:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1001},
                {"timestamp": "2022-01-03 19:00:00", "open": 1.1001, "high": 1.1004, "low": 1.0999, "close": 1.1002},
                {"timestamp": "2022-01-03 19:15:00", "open": 1.1002, "high": 1.1003, "low": 1.1000, "close": 1.1001},
            ]
        )

    def _ambiguous_frame(self) -> pd.DataFrame:
        return make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1001, "atr14": 0.0005},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1000, "high": 1.1012, "low": 1.0988, "close": 1.1005, "atr14": 0.0005},
                {"timestamp": "2022-01-03 11:30:00", "open": 1.1005, "high": 1.1006, "low": 1.1003, "close": 1.1004, "atr14": 0.0005},
            ]
        )

    def test_baseline_normal_mode_matches_existing_behavior(self) -> None:
        frame = self._target_frame()
        params = {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0}
        baseline = run_backtest(LongSignalStrategy, frame, params, self._base_config(), self.news_block, False)
        normal = run_backtest(LongSignalStrategy, frame, params, self._base_config(execution_mode="normal_mode"), self.news_block, False)
        self.assertEqual(len(baseline.trades), len(normal.trades))
        self.assertAlmostEqual(float(baseline.trades.iloc[0]["entry_price"]), float(normal.trades.iloc[0]["entry_price"]), places=8)
        self.assertAlmostEqual(float(baseline.trades.iloc[0]["exit_price"]), float(normal.trades.iloc[0]["exit_price"]), places=8)
        self.assertAlmostEqual(float(baseline.trades.iloc[0]["pnl_usd"]), float(normal.trades.iloc[0]["pnl_usd"]), places=8)

    def test_conservative_mode_marks_trade_and_is_more_expensive(self) -> None:
        frame = self._target_frame()
        params = {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0}
        normal = run_backtest(LongSignalStrategy, frame, params, self._base_config(execution_mode="normal_mode"), self.news_block, False)
        conservative = run_backtest(LongSignalStrategy, frame, params, self._base_config(execution_mode="conservative_mode"), self.news_block, False)
        normal_trade = normal.trades.iloc[0]
        conservative_trade = conservative.trades.iloc[0]
        self.assertEqual(conservative_trade["execution_mode_used"], "conservative_mode")
        self.assertEqual(conservative_trade["cost_profile_used"], "stress")
        self.assertEqual(conservative_trade["intrabar_policy_used"], "conservative")
        self.assertGreater(float(conservative_trade["spread_applied"]), float(normal_trade["spread_applied"]))
        self.assertGreater(float(conservative_trade["slippage_applied"]), float(normal_trade["slippage_applied"]))

    def test_spread_fields_are_logged_per_trade(self) -> None:
        frame = self._target_frame()
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            self._base_config(assumed_spread_pips=1.0, slippage_pips=0.0, commission_per_lot_roundturn_usd=0.0),
            self.news_block,
            False,
        )
        trade = result.trades.iloc[0]
        self.assertAlmostEqual(float(trade["entry_spread_pips"]), 1.0, places=8)
        self.assertAlmostEqual(float(trade["exit_spread_pips"]), 1.0, places=8)
        self.assertAlmostEqual(float(trade["spread_applied"]), 2.0, places=8)

    def test_slippage_fields_are_logged_per_trade(self) -> None:
        frame = self._target_frame()
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            self._base_config(assumed_spread_pips=0.0, slippage_pips=0.5, commission_per_lot_roundturn_usd=0.0),
            self.news_block,
            False,
        )
        trade = result.trades.iloc[0]
        self.assertAlmostEqual(float(trade["entry_slippage_pips"]), 0.5, places=8)
        self.assertAlmostEqual(float(trade["exit_slippage_pips"]), 0.5, places=8)
        self.assertAlmostEqual(float(trade["slippage_applied"]), 1.0, places=8)

    def test_commission_is_separated_from_spread_and_slippage(self) -> None:
        frame = self._target_frame()
        params = {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0}
        no_commission = run_backtest(
            LongSignalStrategy,
            frame,
            params,
            self._base_config(commission_per_lot_roundturn_usd=0.0),
            self.news_block,
            False,
        )
        with_commission = run_backtest(
            LongSignalStrategy,
            frame,
            params,
            self._base_config(commission_per_lot_roundturn_usd=7.0),
            self.news_block,
            False,
        )
        trade_zero = no_commission.trades.iloc[0]
        trade_commission = with_commission.trades.iloc[0]
        self.assertAlmostEqual(float(trade_zero["spread_applied"]), float(trade_commission["spread_applied"]), places=8)
        self.assertAlmostEqual(float(trade_zero["slippage_applied"]), float(trade_commission["slippage_applied"]), places=8)
        self.assertGreater(float(trade_commission["commission_applied"]), 0.0)
        self.assertAlmostEqual(float(trade_commission["commission_applied"]), float(trade_commission["commission_usd"]), places=8)

    def test_intrabar_policy_standard_respects_priority(self) -> None:
        frame = self._ambiguous_frame()
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            self._base_config(
                assumed_spread_pips=0.0,
                slippage_pips=0.0,
                commission_per_lot_roundturn_usd=0.0,
                intrabar_exit_priority="target_first",
                intrabar_policy="standard",
            ),
            self.news_block,
            False,
        )
        trade = result.trades.iloc[0]
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertTrue(bool(trade["intrabar_ambiguity_flag"]))

    def test_intrabar_policy_conservative_overrides_priority(self) -> None:
        frame = self._ambiguous_frame()
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            self._base_config(
                assumed_spread_pips=0.0,
                slippage_pips=0.0,
                commission_per_lot_roundturn_usd=0.0,
                intrabar_exit_priority="target_first",
                intrabar_policy="conservative",
                execution_mode="conservative_mode",
            ),
            self.news_block,
            False,
        )
        trade = result.trades.iloc[0]
        self.assertEqual(trade["exit_reason"], "stop_loss")
        self.assertTrue(bool(trade["intrabar_ambiguity_flag"]))

    def test_forced_close_is_flagged_in_both_modes(self) -> None:
        frame = self._forced_close_frame()
        params = {"signal_times": ["18:45"], "direction": "long", "stop_atr": 10.0, "target_rr": 10.0, "session_name": "light_fixed"}
        for execution_mode in ("normal_mode", "conservative_mode"):
            result = run_backtest(LongSignalStrategy, frame, params, self._base_config(execution_mode=execution_mode), self.news_block, False)
            trade = result.trades.iloc[0]
            self.assertEqual(trade["exit_reason"], "forced_session_close")
            self.assertTrue(bool(trade["forced_close_flag"]))

    def test_final_close_is_flagged_in_both_modes(self) -> None:
        frame = self._target_frame()
        params = {"signal_times": ["11:00"], "direction": "long", "stop_atr": 10.0, "target_rr": 10.0, "session_name": "all_day"}
        for execution_mode in ("normal_mode", "conservative_mode"):
            result = run_backtest(LongSignalStrategy, frame, params, self._base_config(execution_mode=execution_mode), self.news_block, False)
            trade = result.trades.iloc[0]
            self.assertEqual(trade["exit_reason"], "final_bar_close")
            self.assertFalse(bool(trade["forced_close_flag"]))

    def test_trade_export_contains_level2_execution_fields(self) -> None:
        frame = self._target_frame()
        result = run_backtest(
            LongSignalStrategy,
            frame,
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            self._base_config(execution_mode="conservative_mode"),
            self.news_block,
            False,
        )
        exported = build_trades_export(result.trades)
        required_columns = {
            "entry_side",
            "signal_time_ny",
            "fill_time_ny",
            "spread_applied",
            "slippage_applied",
            "commission_applied",
            "forced_close_flag",
            "intrabar_ambiguity_flag",
            "execution_mode_used",
            "intrabar_policy_used",
            "cost_profile_used",
        }
        self.assertTrue(required_columns.issubset(set(exported.columns)))
        self.assertEqual(exported.iloc[0]["execution_mode_used"], "conservative_mode")


if __name__ == "__main__":
    unittest.main()
