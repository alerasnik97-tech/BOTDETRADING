from __future__ import annotations

import builtins
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig, NY_TZ
from research_lab.engine import D5_TELEMETRY_VERSION, run_backtest


def make_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2022-01-03 10:45:00", tz=NY_TZ),
            pd.Timestamp("2022-01-03 11:00:00", tz=NY_TZ),
            pd.Timestamp("2022-01-03 11:15:00", tz=NY_TZ),
            pd.Timestamp("2022-01-03 11:30:00", tz=NY_TZ),
        ]
    )
    return pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1000, 1.1010],
            "high": [1.1002, 1.1002, 1.1015, 1.1012],
            "low": [1.0998, 1.0998, 1.0995, 1.1006],
            "close": [1.1000, 1.1000, 1.1010, 1.1009],
            "atr14": [0.0010, 0.0010, 0.0010, 0.0010],
            "range_atr": [0.5, 0.5, 0.5, 0.5],
        },
        index=index,
    )


class CountingTelemetryStrategy:
    NAME = "d5_synthetic_counting_strategy"
    WARMUP_BARS = 0
    calls: list[tuple[int, pd.Timestamp, dict]] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    @classmethod
    def generate_signal(cls, frame: pd.DataFrame, i: int, params: dict) -> dict | None:
        cls.calls.append((i, frame.index[i], params))
        if frame.index[i].strftime("%H:%M") != params.get("signal_time", "11:00"):
            return None
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": 1.0,
            "target_rr": 1.0,
            "break_even_at_r": None,
            "trailing_atr": False,
            "session_name": "all_day",
        }


class D5CoreTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        CountingTelemetryStrategy.reset()

    def _run_synthetic_trade(self, *, config: EngineConfig | None = None):
        frame = make_frame()
        return run_backtest(
            CountingTelemetryStrategy,
            frame,
            {"signal_time": "11:00", "session_name": "all_day"},
            config
            or EngineConfig(
                pair="EURUSD",
                risk_pct=0.5,
                assumed_spread_pips=0.0,
                max_spread_pips=10.0,
                slippage_pips=0.0,
                commission_per_lot_roundturn_usd=0.0,
                max_trades_per_day=2,
            ),
            np.zeros(len(frame), dtype=bool),
            False,
        )

    def _trade(self):
        result = self._run_synthetic_trade()
        self.assertEqual(len(result.trades), 1)
        return result.trades.iloc[0]

    def test_d5_telemetry_additive_fields_present(self) -> None:
        trade = self._trade()
        required = {
            "telemetry_version",
            "telemetry_behavior_neutral",
            "net_r",
            "gross_r",
            "gross_r_available",
            "sl_pips",
            "sl_pips_available",
            "risk_pips",
            "risk_distance_price",
            "initial_risk_distance",
            "risk_usd",
            "stop_price",
            "initial_stop_price",
            "final_stop_price",
            "entry_spread_pips",
            "entry_slippage_pips",
            "exit_slippage_pips",
            "entry_commission_usd",
            "exit_commission_usd",
            "commission_total_usd",
            "cost_breakdown_r_available",
        }
        self.assertTrue(required.issubset(set(trade.index)))
        self.assertEqual(trade["telemetry_version"], D5_TELEMETRY_VERSION)
        self.assertEqual(bool(trade["telemetry_behavior_neutral"]), True)

    def test_d5_telemetry_existing_trade_fields_unchanged(self) -> None:
        trade = self._trade()
        self.assertEqual(trade["strategy_name"], CountingTelemetryStrategy.NAME)
        self.assertEqual(trade["direction"], "long")
        self.assertAlmostEqual(float(trade["signal_price"]), 1.1000, places=8)
        self.assertAlmostEqual(float(trade["entry_price"]), 1.10012, places=8)
        self.assertAlmostEqual(float(trade["exit_price"]), 1.10124, places=8)
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertAlmostEqual(float(trade["lots"]), 4.4642857143, places=8)
        self.assertEqual(str(trade["session_date"]), "2022-01-03")

    def test_d5_telemetry_does_not_change_pnl(self) -> None:
        trade = self._trade()
        self.assertAlmostEqual(float(trade["pnl_usd"]), 500.0, places=8)
        self.assertAlmostEqual(float(trade["pnl_r"]), 1.0, places=8)
        self.assertAlmostEqual(float(trade["net_r"]), float(trade["pnl_r"]), places=12)

    def test_d5_telemetry_does_not_change_entry_exit_times(self) -> None:
        trade = self._trade()
        entry_time = pd.to_datetime(trade["entry_time"], utc=True).tz_convert(NY_TZ)
        exit_time = pd.to_datetime(trade["exit_time"], utc=True).tz_convert(NY_TZ)
        self.assertEqual(entry_time.strftime("%Y-%m-%d %H:%M"), "2022-01-03 11:15")
        self.assertEqual(exit_time.strftime("%Y-%m-%d %H:%M"), "2022-01-03 11:15")

    def test_d5_telemetry_does_not_change_trade_count(self) -> None:
        result = self._run_synthetic_trade()
        self.assertEqual(len(result.trades), 1)

    def test_d5_telemetry_no_strategy_call_change(self) -> None:
        self._run_synthetic_trade()
        self.assertEqual([call[0] for call in CountingTelemetryStrategy.calls], [0, 1])
        self.assertTrue(all(call[2]["signal_time"] == "11:00" for call in CountingTelemetryStrategy.calls))

    def test_d5_sl_pips_matches_initial_stop_distance_if_available(self) -> None:
        trade = self._trade()
        self.assertEqual(bool(trade["sl_pips_available"]), True)
        self.assertAlmostEqual(float(trade["initial_risk_distance"]), 0.00112, places=8)
        self.assertAlmostEqual(float(trade["risk_distance_price"]), 0.00112, places=8)
        self.assertAlmostEqual(float(trade["sl_pips"]), 11.2, places=8)
        self.assertAlmostEqual(float(trade["risk_pips"]), 11.2, places=8)
        self.assertAlmostEqual(float(trade["stop_price"]), 1.0990, places=8)

    def test_d5_gross_r_not_faked_if_unavailable(self) -> None:
        trade = self._trade()
        self.assertEqual(bool(trade["gross_r_available"]), False)
        self.assertTrue(pd.isna(trade["gross_r"]))
        self.assertEqual(trade["gross_r_reason"], "not_available_without_explicit_pre_cost_pnl_source")
        self.assertEqual(bool(trade["cost_breakdown_r_available"]), False)
        self.assertTrue(pd.isna(trade["cost_total_r"]))

    def test_d5_telemetry_version_present(self) -> None:
        trade = self._trade()
        self.assertEqual(trade["telemetry_version"], "d5_core_telemetry_v1")
        self.assertEqual(bool(trade["telemetry_behavior_neutral"]), True)

    def test_no_real_data_paths_used(self) -> None:
        forbidden_fragments = (
            "raw",
            "tick",
            "ticks",
            "parquet",
            "validation",
            "holdout",
            "2025",
            "2026",
        )
        opened: list[str] = []
        real_open = builtins.open
        real_path_open = Path.open

        def guarded_open(file, *args, **kwargs):
            path = str(file).replace("\\", "/").lower()
            opened.append(path)
            self.assertFalse(any(fragment in path for fragment in forbidden_fragments), path)
            return real_open(file, *args, **kwargs)

        def guarded_path_open(path_self, *args, **kwargs):
            path = str(path_self).replace("\\", "/").lower()
            opened.append(path)
            self.assertFalse(any(fragment in path for fragment in forbidden_fragments), path)
            return real_path_open(path_self, *args, **kwargs)

        with patch("builtins.open", guarded_open), patch.object(Path, "open", guarded_path_open):
            result = self._run_synthetic_trade()

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
