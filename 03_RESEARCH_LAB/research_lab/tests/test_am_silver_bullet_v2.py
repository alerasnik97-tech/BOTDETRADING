from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_lab.config import NY_TZ
from research_lab.strategies import am_silver_bullet_ny_v2


class TestAmSilverBulletV2(unittest.TestCase):
    def setUp(self) -> None:
        times = pd.date_range("2024-01-03 09:55", "2024-01-03 10:10", freq="1min", tz=NY_TZ)
        self.frame = pd.DataFrame(
            index=times,
            data={
                "open": 1.1000,
                "high": 1.1002,
                "low": 1.0998,
                "close": 1.1000,
                "atr14": 0.0004,
                "range_atr": 0.8,
                "bullish_fvg": False,
                "bearish_fvg": False,
                "bullish_fvg_mid": np.nan,
                "bearish_fvg_mid": np.nan,
                "bullish_fvg_size_pips": np.nan,
                "bearish_fvg_size_pips": np.nan,
                "bullish_choch": False,
                "bearish_choch": False,
                "ctx_m5_sb_anchor_high": 1.1010,
                "ctx_m5_sb_anchor_low": 1.0990,
                "ctx_m5_swept_anchor_high": False,
                "ctx_m5_swept_anchor_low": True,
            },
        )
        self.params = am_silver_bullet_ny_v2.default_params()

    def test_emits_long_limit_when_m1_setup_is_ready(self) -> None:
        choch_bar = pd.Timestamp("2024-01-03 10:05", tz=NY_TZ)
        fvg_bar = pd.Timestamp("2024-01-03 10:06", tz=NY_TZ)
        self.frame.at[choch_bar, "bullish_choch"] = True
        self.frame.at[fvg_bar, "bullish_fvg"] = True
        self.frame.at[fvg_bar, "bullish_fvg_mid"] = 1.0996
        self.frame.at[fvg_bar, "bullish_fvg_size_pips"] = 1.2

        idx = self.frame.index.get_loc(fvg_bar)
        signal = am_silver_bullet_ny_v2.signal(self.frame, idx, self.params)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "long")
        self.assertEqual(signal["entry_mode"], "limit")
        self.assertAlmostEqual(signal["limit_price"], 1.0996)
        self.assertAlmostEqual(signal["stop_price"], 1.0988)

    def test_does_not_reissue_setup_after_midpoint_was_already_traded(self) -> None:
        choch_bar = pd.Timestamp("2024-01-03 10:05", tz=NY_TZ)
        fvg_bar = pd.Timestamp("2024-01-03 10:06", tz=NY_TZ)
        touch_bar = pd.Timestamp("2024-01-03 10:07", tz=NY_TZ)
        eval_bar = pd.Timestamp("2024-01-03 10:08", tz=NY_TZ)
        self.frame.at[choch_bar, "bullish_choch"] = True
        self.frame.at[fvg_bar, "bullish_fvg"] = True
        self.frame.at[fvg_bar, "bullish_fvg_mid"] = 1.0996
        self.frame.at[fvg_bar, "bullish_fvg_size_pips"] = 1.2
        self.frame.at[touch_bar, "low"] = 1.0995
        self.frame.at[touch_bar, "high"] = 1.1001

        idx = self.frame.index.get_loc(eval_bar)
        signal = am_silver_bullet_ny_v2.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)

    def test_requires_m5_sweep_context(self) -> None:
        choch_bar = pd.Timestamp("2024-01-03 10:05", tz=NY_TZ)
        fvg_bar = pd.Timestamp("2024-01-03 10:06", tz=NY_TZ)
        self.frame["ctx_m5_swept_anchor_low"] = False
        self.frame.at[choch_bar, "bullish_choch"] = True
        self.frame.at[fvg_bar, "bullish_fvg"] = True
        self.frame.at[fvg_bar, "bullish_fvg_mid"] = 1.0996
        self.frame.at[fvg_bar, "bullish_fvg_size_pips"] = 1.2

        idx = self.frame.index.get_loc(fvg_bar)
        signal = am_silver_bullet_ny_v2.signal(self.frame, idx, self.params)
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()
