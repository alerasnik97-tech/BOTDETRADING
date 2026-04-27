from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import pandas as pd

from research_lab.strategies import pm_volatility_squeeze_retest_m5


class TestPMVolatilitySqueezeRetestM5(unittest.TestCase):
    def setUp(self) -> None:
        self.base_params = {
            "bb_std": 2.0,
            "kc_mult": 1.5,
            "min_squeeze_bars": 5,
            "breakout_buffer_pips": 0.4,
            "limit_expiry_bars": 4,
            "tp_atr_mult": 0.8,
            "be_at_r": None,
            "max_hold_bars": 8,
            "body_pct_filter": 0.55,
            "expansion_atr_min": 0.9,
        }
        self.columns = [
            "open",
            "high",
            "low",
            "close",
            "atr14",
            "range_atr",
            "bb_upper_20_2_0",
            "bb_lower_20_2_0",
            "bb_upper_20_2_2",
            "bb_lower_20_2_2",
            "kc_upper_20_1_5",
            "kc_lower_20_1_5",
            "kc_upper_20_2_0",
            "kc_lower_20_2_0",
        ]

    def _frame(self, overrides: dict[int, dict[str, float]]) -> pd.DataFrame:
        data = []
        base_time = datetime(2023, 1, 2, 12, 0)
        for i in range(80):
            row = {
                "open": 1.1000,
                "high": 1.1004,
                "low": 1.0996,
                "close": 1.1000,
                "atr14": 0.0010,
                "range_atr": 0.60,
                "bb_upper_20_2_0": 1.1005,
                "bb_lower_20_2_0": 1.0995,
                "bb_upper_20_2_2": 1.1006,
                "bb_lower_20_2_2": 1.0994,
                "kc_upper_20_1_5": 1.1010,
                "kc_lower_20_1_5": 1.0990,
                "kc_upper_20_2_0": 1.1011,
                "kc_lower_20_2_0": 1.0989,
            }
            row.update(overrides.get(i, {}))
            data.append(row)
        frame = pd.DataFrame(data, columns=self.columns)
        frame.index = [base_time + timedelta(minutes=5 * i) for i in range(80)]
        return frame

    def test_valid_long_retest_uses_configured_band_set(self) -> None:
        params = dict(self.base_params)
        params.update({"bb_std": 2.2, "kc_mult": 2.0, "tp_atr_mult": 1.0})
        overrides = {
            15: {"close": 1.1004},
            16: {"close": 1.1005},
            17: {"close": 1.1005},
            18: {"close": 1.1004},
            19: {
                "open": 1.1000,
                "high": 1.1020,
                "low": 1.0997,
                "close": 1.1014,
                "range_atr": 1.20,
                "bb_upper_20_2_2": 1.1008,
                "bb_lower_20_2_2": 1.1000,
                "kc_upper_20_2_0": 1.1012,
                "kc_lower_20_2_0": 1.0996,
            },
            20: {
                "open": 1.1014,
                "high": 1.1016,
                "low": 1.1012,
                "close": 1.1013,
                "bb_upper_20_2_2": 1.1009,
                "bb_lower_20_2_2": 1.1001,
            },
            21: {
                "open": 1.1013,
                "high": 1.1014,
                "low": 1.1007,
                "close": 1.1010,
                "bb_upper_20_2_2": 1.1010,
                "bb_lower_20_2_2": 1.1002,
            },
        }
        frame = self._frame(overrides)
        signal = pm_volatility_squeeze_retest_m5.signal(frame, 21, params)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "long")
        self.assertAlmostEqual(float(signal["signal_price"]), 1.1008, places=6)
        self.assertAlmostEqual(float(signal["target_price"]), 1.1018, places=6)
        self.assertAlmostEqual(float(signal["stop_price"]), 1.1000, places=6)

    def test_target_hit_before_retest_cancels_signal(self) -> None:
        params = dict(self.base_params)
        overrides = {
            15: {"close": 1.1004},
            16: {"close": 1.1005},
            17: {"close": 1.1005},
            18: {"close": 1.1004},
            19: {
                "open": 1.1000,
                "high": 1.1020,
                "low": 1.0997,
                "close": 1.1014,
                "range_atr": 1.20,
                "bb_upper_20_2_0": 1.1008,
                "bb_lower_20_2_0": 1.1000,
            },
            20: {
                "open": 1.1014,
                "high": 1.1017,
                "low": 1.1010,
                "close": 1.1015,
                "bb_upper_20_2_0": 1.1008,
                "bb_lower_20_2_0": 1.1000,
            },
            21: {
                "open": 1.1015,
                "high": 1.1016,
                "low": 1.1007,
                "close": 1.1010,
                "bb_upper_20_2_0": 1.1009,
                "bb_lower_20_2_0": 1.1001,
            },
        }
        frame = self._frame(overrides)
        signal = pm_volatility_squeeze_retest_m5.signal(frame, 21, params)
        self.assertIsNone(signal)

    def test_before_13_ny_returns_none(self) -> None:
        params = dict(self.base_params)
        overrides = {
            8: {"close": 1.1004},
            9: {"close": 1.1005},
            10: {"close": 1.1005},
            11: {"close": 1.1004},
            12: {
                "open": 1.1000,
                "high": 1.1020,
                "low": 1.0997,
                "close": 1.1014,
                "range_atr": 1.20,
                "bb_upper_20_2_0": 1.1008,
                "bb_lower_20_2_0": 1.1000,
            },
            13: {"open": 1.1014, "high": 1.1016, "low": 1.1012, "close": 1.1013},
            14: {"open": 1.1013, "high": 1.1014, "low": 1.1007, "close": 1.1010},
        }
        frame = self._frame(overrides)
        self.assertEqual(frame.index[11].strftime("%H:%M"), "12:55")
        self.assertIsNone(pm_volatility_squeeze_retest_m5.signal(frame, 11, params))


if __name__ == "__main__":
    unittest.main()
