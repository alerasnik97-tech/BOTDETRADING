from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from external_scbi_research_harness.matrix import build_baseline_config, build_variants
from external_scbi_research_harness.reporting import summarize_existing_results
from external_scbi_research_harness.strategy import run_truth_model


def build_h1_frame(*, second_sweep: bool) -> pd.DataFrame:
    index = pd.date_range("2020-01-01 00:00", "2020-01-02 23:00", freq="1h", tz="US/Eastern")
    frame = pd.DataFrame(
        {
            "open": 1.1000,
            "high": 1.1008,
            "low": 1.0992,
            "close": 1.1000,
            "volume": 100,
        },
        index=index,
    )

    previous_day = frame.index.date == pd.Timestamp("2020-01-01").date()
    current_day = frame.index.date == pd.Timestamp("2020-01-02").date()

    frame.loc[previous_day, "high"] = 1.1010
    frame.loc[previous_day, "low"] = 1.0990
    frame.loc[previous_day, "open"] = 1.1000
    frame.loc[previous_day, "close"] = 1.1000

    frame.loc[current_day, "open"] = 1.1004
    frame.loc[current_day, "high"] = 1.1009
    frame.loc[current_day, "low"] = 1.1000
    frame.loc[current_day, "close"] = 1.1004

    frame.loc[pd.Timestamp("2020-01-02 00:00", tz="US/Eastern"), ["open", "high", "low", "close"]] = [
        1.1002,
        1.1005,
        1.0985,
        1.0995,
    ]

    if second_sweep:
        frame.loc[pd.Timestamp("2020-01-02 09:00", tz="US/Eastern"), ["open", "high", "low", "close"]] = [
            1.1008,
            1.1016,
            1.1002,
            1.1007,
        ]

    return frame


def build_m5_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-02 00:00", "2020-01-02 14:00", freq="5min", tz="US/Eastern")
    frame = pd.DataFrame(
        {
            "open": 1.1000,
            "high": 1.1002,
            "low": 1.0998,
            "close": 1.1000,
            "volume": 10,
        },
        index=index,
    )

    frame.loc[pd.Timestamp("2020-01-02 01:00", tz="US/Eastern"), ["open", "high", "low", "close"]] = [
        1.0994,
        1.0998,
        1.0993,
        1.0995,
    ]
    frame.loc[pd.Timestamp("2020-01-02 01:05", tz="US/Eastern"), ["open", "high", "low", "close"]] = [
        1.0996,
        1.0998,
        1.0994,
        1.0997,
    ]
    frame.loc[pd.Timestamp("2020-01-02 01:20", tz="US/Eastern"), ["open", "high", "low", "close"]] = [
        1.1008,
        1.1016,
        1.1007,
        1.1014,
    ]
    return frame


def build_news_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp_ny"] = pd.to_datetime(frame["timestamp_ny"], utc=True).dt.tz_convert("US/Eastern")
    frame.attrs["coverage_start_date"] = str(frame["timestamp_ny"].dt.date.min())
    frame.attrs["coverage_end_date"] = str(frame["timestamp_ny"].dt.date.max())
    return frame


class ExternalScbiResearchHarnessTests(unittest.TestCase):
    def test_baseline_truth_model_executes_trade_and_respects_daily_limit(self) -> None:
        config = build_baseline_config("2020-01-02", "2020-01-02")
        result = run_truth_model(
            config,
            h1=build_h1_frame(second_sweep=True),
            m5=build_m5_frame(),
            news=build_news_frame(
                [
                    {
                        "timestamp_ny": "2020-01-02T15:30:00+00:00",
                        "event_name_normalized": "far_same_day_event",
                    }
                ]
            ),
        )

        self.assertEqual(result["stats"]["trades_executed"], 1)
        self.assertGreaterEqual(result["stats"]["daily_limit_skipped"], 1)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"].iloc[0]["exit_reason"], "tp_hit")
        self.assertEqual(result["trades"].iloc[0]["direction"], "long")

    def test_news_filter_ignores_previous_day_event_like_runner_actual(self) -> None:
        config = build_baseline_config("2020-01-02", "2020-01-02")
        result = run_truth_model(
            config,
            h1=build_h1_frame(second_sweep=False),
            m5=build_m5_frame(),
            news=build_news_frame(
                [
                    {
                        "timestamp_ny": "2020-01-02T04:45:00+00:00",
                        "event_name_normalized": "previous_day_close_event",
                    },
                    {
                        "timestamp_ny": "2020-01-02T13:30:00+00:00",
                        "event_name_normalized": "same_day_far_event",
                    },
                ]
            ),
        )

        self.assertEqual(result["stats"].get("news_blocked", 0), 0)
        self.assertEqual(result["stats"]["trades_executed"], 1)
        self.assertEqual(result["trades"].iloc[0]["level_name"], "pdl")

    def test_summarize_existing_results_parses_truth_model_from_csv_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "research_matrix_results.csv"
            output_dir = tmp_path / "summary"

            pd.DataFrame(
                [
                    {
                        "variant_id": "baseline_truth_model",
                        "truth_model": "True",
                        "sample_size": 10,
                        "pf": 1.2,
                        "expectancy": 0.2,
                        "max_drawdown": -2.0,
                        "year_positive_ratio": 0.5,
                        "yearly_total_r_std": 1.0,
                    },
                    {
                        "variant_id": "variant_better",
                        "truth_model": "False",
                        "sample_size": 20,
                        "pf": 1.8,
                        "expectancy": 0.4,
                        "max_drawdown": -1.0,
                        "year_positive_ratio": 1.0,
                        "yearly_total_r_std": 0.5,
                    },
                ]
            ).to_csv(csv_path, index=False)

            files = summarize_existing_results(csv_path, output_dir=output_dir, profile="test_profile")
            summary_payload = json.loads(files["research_summary_json"].read_text(encoding="utf-8"))

            self.assertEqual(summary_payload["baseline_variant"]["variant_id"], "baseline_truth_model")
            self.assertTrue(files["research_top_variants_csv"].exists())
            self.assertTrue(files["research_baseline_vs_variants_md"].exists())

    def test_axis_scan_builds_real_variants_beyond_baseline(self) -> None:
        variants = build_variants(start_date="2020-01-01", end_date="2025-12-31", profile="axis_scan")
        variant_ids = {variant.variant_id for variant in variants}

        self.assertGreater(len(variants), 1)
        self.assertIn("baseline_truth_model", variant_ids)
        self.assertEqual(sum(1 for variant in variants if variant.truth_model), 1)


if __name__ == "__main__":
    unittest.main()
