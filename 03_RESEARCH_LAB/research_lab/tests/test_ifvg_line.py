from __future__ import annotations

import unittest

import pandas as pd

from research_lab.config import NY_TZ
from research_lab.ict_primitives import add_displacement_metrics, add_fvg_columns, add_pivot_structure_columns, find_recent_ifvg_event
from research_lab.strategies import ict_ifvg_repricing_pm


def make_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"], tz=NY_TZ) for item in rows])
    return pd.DataFrame(
        {
            "open": [item["open"] for item in rows],
            "high": [item["high"] for item in rows],
            "low": [item["low"] for item in rows],
            "close": [item["close"] for item in rows],
            "volume": [item.get("volume", 1000) for item in rows],
            "atr14": [item.get("atr14", 0.0008) for item in rows],
            "h1_ema50": [item.get("h1_ema50", 1.1000) for item in rows],
            "h1_ema200": [item.get("h1_ema200", 1.0990) for item in rows],
            "h1_ema200_slope_5": [item.get("h1_ema200_slope_5", 0.0004) for item in rows],
            "close_in_discount_prev_day": [item.get("close_in_discount_prev_day", False) for item in rows],
            "close_in_premium_prev_day": [item.get("close_in_premium_prev_day", False) for item in rows],
            "prev_day_midpoint": [item.get("prev_day_midpoint", 1.1000) for item in rows],
        },
        index=index,
    )


class IfvgLineTests(unittest.TestCase):
    def test_break_close_requires_body_not_wick(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:00:00", "open": 1.1010, "high": 1.1012, "low": 1.1004, "close": 1.1006},
                {"timestamp": "2024-01-03 11:05:00", "open": 1.1006, "high": 1.1008, "low": 1.0990, "close": 1.0992},
                {"timestamp": "2024-01-03 11:10:00", "open": 1.0992, "high": 1.1000, "low": 1.0991, "close": 1.0998},
                {"timestamp": "2024-01-03 11:15:00", "open": 1.0998, "high": 1.1006, "low": 1.0996, "close": 1.1004},
                {"timestamp": "2024-01-03 11:20:00", "open": 1.1004, "high": 1.1005, "low": 1.0989, "close": 1.0991},
                {"timestamp": "2024-01-03 11:25:00", "open": 1.0991, "high": 1.0998, "low": 1.0990, "close": 1.0997},
                {"timestamp": "2024-01-03 11:30:00", "open": 1.0997, "high": 1.1014, "low": 1.0996, "close": 1.0999},
                {"timestamp": "2024-01-03 11:35:00", "open": 1.0999, "high": 1.1016, "low": 1.0998, "close": 1.1013},
            ]
        )
        enriched = add_pivot_structure_columns(frame, left_bars=1, right_bars=1, break_buffer_pips=0.1)
        self.assertFalse(bool(enriched["bullish_break_close"].iloc[6]))
        self.assertTrue(bool(enriched["bullish_break_close"].iloc[7]))

    def test_ifvg_requires_full_close_through_gap(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:00:00", "open": 1.1010, "high": 1.1012, "low": 1.1006, "close": 1.1008},
                {"timestamp": "2024-01-03 11:05:00", "open": 1.1007, "high": 1.1009, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2024-01-03 11:10:00", "open": 1.1000, "high": 1.1001, "low": 1.0997, "close": 1.0998},
                {"timestamp": "2024-01-03 11:15:00", "open": 1.0998, "high": 1.1004, "low": 1.0996, "close": 1.1003},
                {"timestamp": "2024-01-03 11:20:00", "open": 1.1003, "high": 1.1014, "low": 1.1002, "close": 1.1013},
            ]
        )
        enriched = add_fvg_columns(frame)
        enriched = add_displacement_metrics(enriched, recent_lookback=1)
        enriched["bullish_break_close"] = [False, False, False, False, True]
        enriched["bearish_break_close"] = [False] * len(enriched)

        event = find_recent_ifvg_event(
            enriched,
            4,
            direction="long",
            min_fvg_pips=0.5,
            min_fvg_atr=0.05,
            max_fvg_age_bars=6,
            max_inversion_bars=4,
            max_retest_bars=4,
            require_break_close=True,
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.source_fvg_direction, "bearish")
        self.assertEqual(event.inversion_bar_index, 4)

        enriched.loc[enriched.index[4], "close"] = 1.1005
        enriched.loc[enriched.index[4], "bullish_break_close"] = False
        no_event = find_recent_ifvg_event(
            enriched,
            4,
            direction="long",
            min_fvg_pips=0.5,
            min_fvg_atr=0.05,
            max_fvg_age_bars=6,
            max_inversion_bars=4,
            max_retest_bars=4,
            require_break_close=True,
        )
        self.assertIsNone(no_event)

    def test_strategy_emits_long_signal_on_ifvg_retest(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:00:00", "open": 1.1010, "high": 1.1012, "low": 1.1006, "close": 1.1008, "close_in_discount_prev_day": True},
                {"timestamp": "2024-01-03 11:05:00", "open": 1.1007, "high": 1.1009, "low": 1.0998, "close": 1.1000, "close_in_discount_prev_day": True},
                {"timestamp": "2024-01-03 11:10:00", "open": 1.1000, "high": 1.1001, "low": 1.0997, "close": 1.0998, "close_in_discount_prev_day": True},
                {"timestamp": "2024-01-03 11:15:00", "open": 1.0998, "high": 1.1004, "low": 1.0996, "close": 1.1003, "close_in_discount_prev_day": True},
                {"timestamp": "2024-01-03 11:20:00", "open": 1.1003, "high": 1.1014, "low": 1.1002, "close": 1.1013, "atr14": 0.0008, "close_in_discount_prev_day": True},
                {"timestamp": "2024-01-03 11:25:00", "open": 1.1004, "high": 1.1010, "low": 1.1003, "close": 1.1009, "atr14": 0.0008, "close_in_discount_prev_day": True},
            ]
        )
        enriched = add_fvg_columns(frame)
        enriched = add_displacement_metrics(enriched, recent_lookback=1)
        enriched["bullish_break_close"] = [False, False, False, False, True, False]
        enriched["bearish_break_close"] = [False] * len(enriched)

        signal = ict_ifvg_repricing_pm.signal(enriched, 5, ict_ifvg_repricing_pm.default_params())
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal["direction"], "long")
        self.assertEqual(signal["stop_mode"], "price")


if __name__ == "__main__":
    unittest.main()
