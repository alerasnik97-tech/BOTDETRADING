from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_lab.config import NY_TZ
from research_lab.eurusd_ltf_objective_entry_replacement_ecb_autopilot import (
    _extreme_candle,
    _h1_sweep_candidate,
    annotate_ecb_frame,
)
from research_lab.strategies import eurusd_ltf_objective_entry_replacement_ecb as strategy_module


class TestEurusdLtfObjectiveEntryReplacementEcb(unittest.TestCase):
    def test_h1_short_sweep_candidate_rejects_breakout_body_outside(self) -> None:
        candidate = _h1_sweep_candidate(
            pd.Series(
                {
                    "open": 1.1008,
                    "high": 1.1010,
                    "low": 1.0998,
                    "close": 1.0999,
                    "prev_day_high": 1.1000,
                    "asia_complete": True,
                    "london_complete": True,
                }
            ),
            direction="short",
        )
        self.assertIsNone(candidate)

    def test_h1_short_sweep_candidate_accepts_rejection(self) -> None:
        candidate = _h1_sweep_candidate(
            pd.Series(
                {
                    "open": 1.0998,
                    "high": 1.1008,
                    "low": 1.0992,
                    "close": 1.0994,
                    "prev_day_high": 1.1000,
                    "asia_complete": True,
                    "london_complete": True,
                }
            ),
            direction="short",
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["direction"], "short")
        self.assertEqual(candidate["level_name"], "prev_day_high")
        self.assertAlmostEqual(float(candidate["sweep_price"]), 1.1008)

    def test_extreme_candle_returns_unique_high_and_opposite_low(self) -> None:
        index = pd.date_range("2024-01-03 09:03", periods=4, freq="3min", tz=NY_TZ)
        frame = pd.DataFrame(
            {
                "open": [1.0998, 1.1000, 1.1002, 1.1001],
                "high": [1.1002, 1.1006, 1.1009, 1.1004],
                "low": [1.0996, 1.0999, 1.1001, 1.0998],
                "close": [1.1000, 1.1003, 1.1002, 1.1000],
            },
            index=index,
        )
        extreme = _extreme_candle(frame, direction="short", level_price=1.1000)
        self.assertIsNotNone(extreme)
        assert extreme is not None
        self.assertEqual(extreme["timestamp"], index[2])
        self.assertAlmostEqual(float(extreme["entry_trigger"]), 1.1001)

    def test_annotation_emits_short_signal_and_strategy_returns_stop_order(self) -> None:
        m3_index = pd.date_range("2024-01-03 09:03", periods=20, freq="3min", tz=NY_TZ)
        m3_frame = pd.DataFrame(
            {
                "open": np.full(len(m3_index), 1.0998),
                "high": np.full(len(m3_index), 1.1001),
                "low": np.full(len(m3_index), 1.0995),
                "close": np.full(len(m3_index), 1.0999),
            },
            index=m3_index,
        )
        m3_frame.at[pd.Timestamp("2024-01-03 09:48", tz=NY_TZ), "high"] = 1.1009
        m3_frame.at[pd.Timestamp("2024-01-03 09:48", tz=NY_TZ), "low"] = 1.1002
        m3_frame.at[pd.Timestamp("2024-01-03 09:48", tz=NY_TZ), "close"] = 1.1004
        m3_frame.at[pd.Timestamp("2024-01-03 10:00", tz=NY_TZ), "close"] = 1.1003
        h1_index = pd.DatetimeIndex([pd.Timestamp("2024-01-03 10:00", tz=NY_TZ)])
        h1_frame = pd.DataFrame(
            {
                "open": [1.0998],
                "high": [1.1008],
                "low": [1.0992],
                "close": [1.0994],
                "prev_day_high": [1.1000],
                "prev_day_low": [np.nan],
                "asia_high": [np.nan],
                "asia_low": [np.nan],
                "london_high": [np.nan],
                "london_low": [np.nan],
                "asia_complete": [True],
                "london_complete": [True],
            },
            index=h1_index,
        )

        annotated, _signal_log = annotate_ecb_frame(m3_frame, h1_frame)
        signal_ts = pd.Timestamp("2024-01-03 10:00", tz=NY_TZ)
        self.assertTrue(bool(annotated.at[signal_ts, "ecb_signal"]))
        self.assertEqual(str(annotated.at[signal_ts, "ecb_direction"]), "short")
        self.assertAlmostEqual(float(annotated.at[signal_ts, "ecb_stop_entry_price"]), 1.1002)
        self.assertAlmostEqual(float(annotated.at[signal_ts, "ecb_stop_price"]), 1.1009)

        params = strategy_module.default_params()
        signal = strategy_module.signal(annotated, annotated.index.get_loc(signal_ts), params)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal["entry_mode"], "stop")
        self.assertAlmostEqual(float(signal["stop_entry_price"]), 1.1002)
        self.assertAlmostEqual(float(signal["stop_price"]), 1.1009)
        self.assertAlmostEqual(float(signal["target_rr"]), 1.5)


if __name__ == "__main__":
    unittest.main()
