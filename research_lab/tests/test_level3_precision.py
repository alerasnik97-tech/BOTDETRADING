from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab.config import DEFAULT_NEWS_FILE, DEFAULT_NEWS_SUMMARY_FILE, DEFAULT_RAW_NEWS_FILE, EngineConfig, NY_TZ, NewsConfig
from research_lab.engine import run_backtest
from research_lab.news_filter import load_news_events, load_news_summary
from research_lab.report import build_trades_export


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


class LongSignalStrategy:
    NAME = "long_signal_level3"
    WARMUP_BARS = 0

    @staticmethod
    def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
        if frame.index[i].strftime("%H:%M") in set(params.get("signal_times", [])):
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


class Level3PrecisionTests(unittest.TestCase):
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

    def test_high_precision_mode_uses_precision_profile(self) -> None:
        result = run_backtest(
            LongSignalStrategy,
            self._target_frame(),
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            self._base_config(execution_mode="high_precision_mode"),
            self.news_block,
            False,
        )
        trade = result.trades.iloc[0]
        self.assertEqual(trade["execution_mode_used"], "high_precision_mode")
        self.assertEqual(trade["cost_profile_used"], "precision")
        self.assertEqual(trade["intrabar_policy_used"], "standard")

    def test_high_precision_mode_logs_signal_and_fill_prices(self) -> None:
        result = run_backtest(
            LongSignalStrategy,
            self._target_frame(),
            {"signal_times": ["11:00"], "direction": "long", "stop_atr": 1.0, "target_rr": 1.0},
            self._base_config(execution_mode="high_precision_mode"),
            self.news_block,
            False,
        )
        trade = result.trades.iloc[0]
        exported = build_trades_export(result.trades)
        self.assertAlmostEqual(float(trade["signal_price"]), 1.1001, places=6)
        self.assertAlmostEqual(float(trade["fill_price"]), float(trade["entry_price"]), places=8)
        self.assertIn("signal_price", exported.columns)
        self.assertIn("fill_price", exported.columns)
        self.assertIn("exit_signal_price", exported.columns)
        self.assertIn("exit_fill_price", exported.columns)
        self.assertIn("price_source_used", exported.columns)

    def test_high_precision_mode_forced_close_is_more_expensive_than_normal(self) -> None:
        params = {"signal_times": ["18:45"], "direction": "long", "stop_atr": 10.0, "target_rr": 10.0, "session_name": "light_fixed"}
        normal = run_backtest(
            LongSignalStrategy,
            self._forced_close_frame(),
            params,
            self._base_config(execution_mode="normal_mode"),
            self.news_block,
            False,
        )
        precision = run_backtest(
            LongSignalStrategy,
            self._forced_close_frame(),
            params,
            self._base_config(execution_mode="high_precision_mode"),
            self.news_block,
            False,
        )
        self.assertGreater(float(precision.trades.iloc[0]["slippage_applied"]), float(normal.trades.iloc[0]["slippage_applied"]))

    def test_real_validated_news_dataset_is_operational_when_enabled(self) -> None:
        result = load_news_events(
            "EURUSD",
            NewsConfig(
                enabled=True,
                file_path=Path(DEFAULT_NEWS_FILE),
                raw_file_path=Path(DEFAULT_RAW_NEWS_FILE),
                source_approved=True,
                pre_minutes=15,
                post_minutes=15,
                currencies=("USD", "EUR"),
            ),
        )
        self.assertTrue(result.enabled)
        self.assertGreater(result.approved_rows, 0)
        self.assertEqual(result.source_name, "forex_factory_fixed_schedule_validated")

    def test_real_news_summary_reports_operational_approval(self) -> None:
        summary = load_news_summary(Path(DEFAULT_NEWS_SUMMARY_FILE))
        self.assertEqual(summary.get("module_verdict"), "APPROVED_OPERATIONAL")
        self.assertTrue(bool(summary.get("source_approved")))
        key_status = {row["event_name_normalized"]: row["status"] for row in summary.get("key_event_validation", [])}
        self.assertIn(key_status.get("gdp q/q"), {"PASS_ALIAS_FAMILY", "PASS"})
        self.assertIn(key_status.get("ppi y/y"), {"SOURCE_ABSENT", "PASS"})

    def test_real_news_summary_json_is_valid_json(self) -> None:
        payload = json.loads(Path(DEFAULT_NEWS_SUMMARY_FILE).read_text(encoding="utf-8"))
        self.assertIn("operational_source_verdict", payload)
        self.assertIn("key_event_validation", payload)


if __name__ == "__main__":
    unittest.main()
