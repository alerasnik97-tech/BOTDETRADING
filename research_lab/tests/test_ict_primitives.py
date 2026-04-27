from __future__ import annotations

import unittest

import pandas as pd

from research_lab.config import NY_TZ
from research_lab.data_loader import prepare_common_frame
from research_lab.ict_primitives import (
    add_displacement_metrics,
    add_equal_high_low_columns,
    add_fvg_columns,
    add_pivot_structure_columns,
    add_premium_discount_columns,
    add_previous_period_levels,
    add_session_level_aliases,
    bullish_displacement,
    bearish_displacement,
    find_recent_sweep_event,
)


def make_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"], tz=NY_TZ) for item in rows])
    return pd.DataFrame(
        {
            "open": [item["open"] for item in rows],
            "high": [item["high"] for item in rows],
            "low": [item["low"] for item in rows],
            "close": [item["close"] for item in rows],
            "volume": [item.get("volume", 1000) for item in rows],
            "atr14": [item.get("atr14", 0.0010) for item in rows],
            "h1_ema50": [item.get("h1_ema50", 1.1000) for item in rows],
            "h1_ema200": [item.get("h1_ema200", 1.0990) for item in rows],
            "h1_ema200_slope_5": [item.get("h1_ema200_slope_5", 0.0010) for item in rows],
        },
        index=index,
    )


class IctPrimitivesTests(unittest.TestCase):
    def test_session_levels_and_previous_periods_are_computed(self) -> None:
        rows = []
        start = pd.Timestamp("2024-01-01 00:00:00", tz=NY_TZ)
        price = 1.1000
        for day in range(10):
            for minutes in range(0, 24 * 60, 60):
                ts = start + pd.Timedelta(days=day, minutes=minutes)
                rows.append(
                    {
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "open": price,
                        "high": price + (0.0005 if ts.hour == 6 else 0.0002),
                        "low": price - (0.0005 if ts.hour == 2 else 0.0002),
                        "close": price + 0.0001,
                        "volume": 1000,
                    }
                )
                price += 0.00005

        frame = make_frame(rows)
        frame["session_range_high_19_00_03_00"] = 1.2000
        frame["session_range_low_19_00_03_00"] = 1.1900
        frame["session_range_complete_19_00_03_00"] = True
        frame["session_range_high_03_00_07_00"] = 1.2100
        frame["session_range_low_03_00_07_00"] = 1.1850
        frame["session_range_complete_03_00_07_00"] = True

        enriched = add_session_level_aliases(frame)
        enriched = add_previous_period_levels(enriched)

        row = enriched.loc[pd.Timestamp("2024-01-08 12:00:00", tz=NY_TZ)]
        self.assertEqual(float(row["asia_high"]), 1.2000)
        self.assertEqual(float(row["london_low"]), 1.1850)
        self.assertTrue(pd.notna(row["prev_day_high"]))
        self.assertTrue(pd.notna(row["prev_week_high"]))

    def test_exact_manual_session_windows_handle_midnight_and_completion(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-01 18:00:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000},
                {"timestamp": "2024-01-01 19:00:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005},
                {"timestamp": "2024-01-01 22:00:00", "open": 1.1005, "high": 1.1040, "low": 1.0980, "close": 1.1030},
                {"timestamp": "2024-01-02 02:00:00", "open": 1.1030, "high": 1.1032, "low": 1.0970, "close": 1.0985},
                {"timestamp": "2024-01-02 04:00:00", "open": 1.0985, "high": 1.1050, "low": 1.1000, "close": 1.1040},
                {"timestamp": "2024-01-02 06:00:00", "open": 1.1040, "high": 1.1060, "low": 1.0995, "close": 1.1000},
                {"timestamp": "2024-01-02 07:00:00", "open": 1.1000, "high": 1.1004, "low": 1.0997, "close": 1.1001},
            ]
        )

        enriched = add_session_level_aliases(frame)

        pre_complete_row = enriched.loc[pd.Timestamp("2024-01-02 02:00:00", tz=NY_TZ)]
        asia_complete_row = enriched.loc[pd.Timestamp("2024-01-02 04:00:00", tz=NY_TZ)]
        london_complete_row = enriched.loc[pd.Timestamp("2024-01-02 07:00:00", tz=NY_TZ)]

        self.assertTrue(pd.isna(pre_complete_row["asia_high"]))
        self.assertFalse(bool(pre_complete_row["asia_complete"]))
        self.assertAlmostEqual(float(asia_complete_row["asia_high"]), 1.1040, places=5)
        self.assertAlmostEqual(float(asia_complete_row["asia_low"]), 1.0970, places=5)
        self.assertTrue(bool(asia_complete_row["asia_complete"]))
        self.assertTrue(pd.isna(asia_complete_row["london_high"]))
        self.assertFalse(bool(asia_complete_row["london_complete"]))
        self.assertAlmostEqual(float(london_complete_row["london_high"]), 1.1060, places=5)
        self.assertAlmostEqual(float(london_complete_row["london_low"]), 1.0995, places=5)
        self.assertTrue(bool(london_complete_row["london_complete"]))

    def test_recent_sweep_detector_marks_same_bar_reclaim(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:30:00", "open": 1.1000, "high": 1.1004, "low": 1.0997, "close": 1.1001},
                {"timestamp": "2024-01-03 11:35:00", "open": 1.1001, "high": 1.1011, "low": 1.0999, "close": 1.0998},
            ]
        )
        frame["prev_day_high"] = [1.1005, 1.1005]
        event = find_recent_sweep_event(frame, 1, direction="short", min_penetration_pips=0.5, max_age_bars=0, level_columns=("prev_day_high",))
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.level_name, "prev_day_high")
        self.assertEqual(event.direction, "short")

    def test_displacement_metrics_require_body_close_and_expansion(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:00:00", "open": 1.1000, "high": 1.1004, "low": 1.0998, "close": 1.1001, "atr14": 0.0005},
                {"timestamp": "2024-01-03 11:05:00", "open": 1.1001, "high": 1.1018, "low": 1.1000, "close": 1.1017, "atr14": 0.0006},
                {"timestamp": "2024-01-03 11:10:00", "open": 1.1017, "high": 1.1019, "low": 1.1013, "close": 1.1014, "atr14": 0.0006},
            ]
        )
        enriched = add_displacement_metrics(frame, recent_lookback=1)
        self.assertTrue(
            bullish_displacement(
                enriched,
                1,
                min_body_atr=1.5,
                min_body_fraction=0.7,
                min_close_location=0.75,
                min_range_expansion=2.0,
            )
        )
        self.assertFalse(
            bearish_displacement(
                enriched,
                2,
                min_body_atr=1.5,
                min_body_fraction=0.7,
                max_close_location=0.25,
                min_range_expansion=2.0,
            )
        )

    def test_fvg_detector_builds_strict_three_bar_gap(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:00:00", "open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1002},
                {"timestamp": "2024-01-03 11:05:00", "open": 1.1003, "high": 1.1015, "low": 1.1002, "close": 1.1014},
                {"timestamp": "2024-01-03 11:10:00", "open": 1.1014, "high": 1.1020, "low": 1.1008, "close": 1.1018},
            ]
        )
        enriched = add_fvg_columns(frame)
        row = enriched.iloc[2]
        self.assertTrue(bool(row["bullish_fvg"]))
        self.assertAlmostEqual(float(row["bullish_fvg_bottom"]), 1.1005, places=5)
        self.assertAlmostEqual(float(row["bullish_fvg_top"]), 1.1008, places=5)

    def test_pivot_structure_can_mark_bullish_choch(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:00:00", "open": 1.1010, "high": 1.1012, "low": 1.1004, "close": 1.1006},
                {"timestamp": "2024-01-03 11:05:00", "open": 1.1006, "high": 1.1008, "low": 1.0990, "close": 1.0992},
                {"timestamp": "2024-01-03 11:10:00", "open": 1.0992, "high": 1.1000, "low": 1.0991, "close": 1.0998},
                {"timestamp": "2024-01-03 11:15:00", "open": 1.0998, "high": 1.1006, "low": 1.0996, "close": 1.1004},
                {"timestamp": "2024-01-03 11:20:00", "open": 1.1004, "high": 1.1005, "low": 1.0989, "close": 1.0991},
                {"timestamp": "2024-01-03 11:25:00", "open": 1.0991, "high": 1.0998, "low": 1.0990, "close": 1.0997},
                {"timestamp": "2024-01-03 11:30:00", "open": 1.0997, "high": 1.1015, "low": 1.0996, "close": 1.1013},
                {"timestamp": "2024-01-03 11:35:00", "open": 1.1013, "high": 1.1016, "low": 1.1008, "close": 1.1012},
            ]
        )
        enriched = add_pivot_structure_columns(frame, left_bars=1, right_bars=1, break_buffer_pips=0.1)
        self.assertTrue(bool(enriched["bullish_choch"].iloc[6]))

    def test_equal_highs_use_fixed_tolerance(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 11:00:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1001},
                {"timestamp": "2024-01-03 11:05:00", "open": 1.1001, "high": 1.1010, "low": 1.0999, "close": 1.1002},
                {"timestamp": "2024-01-03 11:10:00", "open": 1.1002, "high": 1.1003, "low": 1.0997, "close": 1.1000},
                {"timestamp": "2024-01-03 11:15:00", "open": 1.1000, "high": 1.10105, "low": 1.0999, "close": 1.1001},
                {"timestamp": "2024-01-03 11:20:00", "open": 1.1001, "high": 1.1004, "low": 1.0998, "close": 1.1000},
            ]
        )
        enriched = add_pivot_structure_columns(frame, left_bars=1, right_bars=1, break_buffer_pips=0.1)
        enriched = add_equal_high_low_columns(enriched, tolerance_pips=1.0, max_separation_bars=10)
        self.assertTrue(bool(enriched["equal_high"].iloc[4]))

    def test_premium_discount_uses_midpoint_of_explicit_range(self) -> None:
        frame = make_frame(
            [
                {"timestamp": "2024-01-03 12:00:00", "open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.0999},
                {"timestamp": "2024-01-03 12:05:00", "open": 1.1000, "high": 1.1004, "low": 1.0999, "close": 1.1003},
            ]
        )
        frame["prev_day_high"] = [1.1020, 1.1020]
        frame["prev_day_low"] = [1.0980, 1.0980]
        enriched = add_premium_discount_columns(frame, "prev_day_high", "prev_day_low", "prev_day")
        self.assertTrue(bool(enriched["close_in_discount_prev_day"].iloc[0]))
        self.assertTrue(bool(enriched["close_in_premium_prev_day"].iloc[1]))

    def test_prepare_common_frame_injects_ict_columns(self) -> None:
        rows = []
        start = pd.Timestamp("2024-01-01 00:00:00", tz=NY_TZ)
        price = 1.1000
        for i in range(900):
            ts = start + pd.Timedelta(minutes=5 * i)
            drift = 0.0002 if i % 17 == 0 else -0.0001 if i % 11 == 0 else 0.00002
            open_price = price
            close_price = price + drift
            high_price = max(open_price, close_price) + 0.00025
            low_price = min(open_price, close_price) - 0.00025
            rows.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": 1000,
                }
            )
            price = close_price

        raw = make_frame(rows)[["open", "high", "low", "close", "volume"]]
        enriched = prepare_common_frame(raw, target_timeframe="M5")
        for column in (
            "asia_high",
            "london_low",
            "session_range_high_19_00_03_00",
            "session_range_high_03_00_07_00",
            "prev_week_high",
            "bullish_fvg",
            "last_confirmed_swing_high",
            "close_in_discount_prev_day",
        ):
            self.assertIn(column, enriched.columns)
