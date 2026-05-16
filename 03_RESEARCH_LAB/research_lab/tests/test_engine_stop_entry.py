from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig, NY_TZ
from research_lab.engine import run_backtest


def make_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"], tz=NY_TZ) for item in rows])
    return pd.DataFrame(
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


class StopEntryStrategy:
    NAME = "stop_entry_strategy"
    WARMUP_BARS = 0

    @staticmethod
    def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
        if frame.index[i].strftime("%H:%M") != params.get("signal_time", "11:00"):
            return None
        return {
            "direction": params.get("direction", "long"),
            "entry_mode": "stop",
            "stop_entry_price": params["stop_entry_price"],
            "stop_mode": "price",
            "stop_price": params["stop_price"],
            "target_mode": "rr",
            "target_rr": params.get("target_rr", 1.0),
            "session_name": "all_day",
        }

    generate_signal = signal


class InvalidStopEntryStrategy:
    NAME = "invalid_stop_entry_strategy"
    WARMUP_BARS = 0

    @staticmethod
    def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
        if frame.index[i].strftime("%H:%M") != "11:00":
            return None
        return {
            "direction": "long",
            "entry_mode": "stop",
            "stop_entry_price": 1.1000,
            "stop_mode": "price",
            "stop_price": 1.0990,
            "target_mode": "rr",
            "target_rr": 1.0,
            "session_name": "all_day",
        }

    generate_signal = signal


class EngineStopEntryTests(unittest.TestCase):
    def test_long_stop_entry_fills_at_trigger_inside_next_bar(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1001, "high": 1.1004, "low": 1.1000, "close": 1.1003},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1004, "high": 1.1009, "low": 1.1002, "close": 1.1008},
                {"timestamp": "2022-01-03 11:30:00", "open": 1.1008, "high": 1.1026, "low": 1.1007, "close": 1.1023},
            ]
        )
        result = run_backtest(
            StopEntryStrategy,
            frame,
            {
                "direction": "long",
                "stop_entry_price": 1.1008,
                "stop_price": 1.0996,
                "target_rr": 1.0,
            },
            EngineConfig(
                pair="EURUSD",
                risk_pct=0.5,
                assumed_spread_pips=1.2,
                max_spread_pips=2.0,
                slippage_pips=0.0,
                commission_per_lot_roundturn_usd=0.0,
                max_trades_per_day=2,
            ),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertAlmostEqual(float(trade["entry_price"]), 1.10092, places=5)
        self.assertEqual(trade["exit_reason"], "take_profit")

    def test_short_stop_entry_uses_gap_open_when_next_bar_opens_through_level(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.0999, "high": 1.1000, "low": 1.0995, "close": 1.0994},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.0990, "high": 1.0991, "low": 1.0988, "close": 1.0989},
                {"timestamp": "2022-01-03 11:30:00", "open": 1.0989, "high": 1.0989, "low": 1.0975, "close": 1.0978},
                {"timestamp": "2022-01-03 11:45:00", "open": 1.0978, "high": 1.0979, "low": 1.0970, "close": 1.0972},
            ]
        )
        result = run_backtest(
            StopEntryStrategy,
            frame,
            {
                "direction": "short",
                "stop_entry_price": 1.0992,
                "stop_price": 1.1004,
                "target_rr": 1.0,
            },
            EngineConfig(
                pair="EURUSD",
                risk_pct=0.5,
                assumed_spread_pips=1.2,
                max_spread_pips=2.0,
                slippage_pips=0.0,
                commission_per_lot_roundturn_usd=0.0,
                max_trades_per_day=2,
            ),
            np.zeros(len(frame), dtype=bool),
            False,
        )
        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertAlmostEqual(float(trade["entry_price"]), 1.0990, places=5)
        self.assertEqual(trade["exit_reason"], "take_profit")

    def test_invalid_stop_entry_direction_is_rejected(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.1001, "high": 1.1002, "low": 1.0999, "close": 1.1001},
                {"timestamp": "2022-01-03 11:15:00", "open": 1.1002, "high": 1.1004, "low": 1.1000, "close": 1.1003},
            ]
        )
        with self.assertRaisesRegex(ValueError, "stop entry"):
            run_backtest(
                InvalidStopEntryStrategy,
                frame,
                {},
                EngineConfig(
                    pair="EURUSD",
                    risk_pct=0.5,
                    assumed_spread_pips=0.0,
                    max_spread_pips=2.0,
                    slippage_pips=0.0,
                    commission_per_lot_roundturn_usd=0.0,
                    max_trades_per_day=2,
                ),
                np.zeros(len(frame), dtype=bool),
                False,
            )


if __name__ == "__main__":
    unittest.main()
