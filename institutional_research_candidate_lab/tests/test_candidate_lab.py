from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from institutional_research_candidate_lab.baseline_truth_model import run_baseline_truth_model
from institutional_research_candidate_lab.candidate_matrix import build_baseline_config, build_candidate_matrix
from institutional_research_candidate_lab.reporting import summarize_existing_results, write_matrix_outputs


def _build_h1_frame(*, second_sweep: bool) -> pd.DataFrame:
    index = pd.date_range("2020-01-01 00:00", "2020-01-02 23:00", freq="1h", tz="US/Eastern")
    frame = pd.DataFrame({"open": 1.1000, "high": 1.1008, "low": 1.0992, "close": 1.1000, "volume": 100}, index=index)
    previous_day = frame.index.date == pd.Timestamp("2020-01-01").date()
    current_day = frame.index.date == pd.Timestamp("2020-01-02").date()
    frame.loc[previous_day, ["open", "high", "low", "close"]] = [1.1000, 1.1010, 1.0990, 1.1000]
    frame.loc[current_day, ["open", "high", "low", "close"]] = [1.1004, 1.1009, 1.1000, 1.1004]
    frame.loc[pd.Timestamp("2020-01-02 00:00", tz="US/Eastern"), ["open", "high", "low", "close"]] = [1.1002, 1.1005, 1.0985, 1.0995]
    if second_sweep:
        frame.loc[pd.Timestamp("2020-01-02 09:00", tz="US/Eastern"), ["open", "high", "low", "close"]] = [1.1008, 1.1016, 1.1002, 1.1007]
    return frame


def _build_m5_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-02 00:00", "2020-01-02 14:00", freq="5min", tz="US/Eastern")
    frame = pd.DataFrame({"open": 1.1000, "high": 1.1002, "low": 1.0998, "close": 1.1000, "volume": 10}, index=index)
    frame.loc[pd.Timestamp("2020-01-02 01:00", tz="US/Eastern"), ["open", "high", "low", "close"]] = [1.0994, 1.0998, 1.0993, 1.0995]
    frame.loc[pd.Timestamp("2020-01-02 01:05", tz="US/Eastern"), ["open", "high", "low", "close"]] = [1.0996, 1.0998, 1.0994, 1.0997]
    frame.loc[pd.Timestamp("2020-01-02 01:20", tz="US/Eastern"), ["open", "high", "low", "close"]] = [1.1008, 1.1016, 1.1007, 1.1014]
    return frame


def _build_news_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp_ny"] = pd.to_datetime(frame["timestamp_ny"], utc=True).dt.tz_convert("US/Eastern")
    frame.attrs["coverage_start_date"] = str(frame["timestamp_ny"].dt.date.min())
    frame.attrs["coverage_end_date"] = str(frame["timestamp_ny"].dt.date.max())
    return frame


class CandidateLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_root = ROOT / "institutional_research_candidate_lab" / "tests" / "_tmp_runtime"
        if cls.tmp_root.exists():
            shutil.rmtree(cls.tmp_root)
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmp_root.exists():
            shutil.rmtree(cls.tmp_root)

    def test_baseline_replica_executes_trade_and_daily_limit(self) -> None:
        config = build_baseline_config("2020-01-02", "2020-01-02")
        result = run_baseline_truth_model(
            config,
            h1=_build_h1_frame(second_sweep=True),
            m5=_build_m5_frame(),
            news=_build_news_frame([{"timestamp_ny": "2020-01-02T15:30:00+00:00", "event_name_normalized": "far_event"}]),
        )
        self.assertEqual(result["stats"]["trades_executed"], 1)
        self.assertGreaterEqual(result["stats"]["daily_limit_skipped"], 1)
        self.assertEqual(result["trades"].iloc[0]["exit_reason"], "tp_hit")

    def test_candidate_matrix_builds_multiple_variants(self) -> None:
        variants = build_candidate_matrix(start_date="2020-01-01", end_date="2025-12-31", profile="axis_scan")
        self.assertGreater(len(variants), 1)
        self.assertEqual(sum(1 for variant in variants if variant.truth_model), 1)

    def test_reporting_writes_outputs(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "variant_id": "baseline_truth_model",
                    "profile_name": "baseline",
                    "truth_model": True,
                    "research_status": "RESEARCH_ONLY",
                    "promotion_status": "NO_PRODUCTION",
                    "experimental_variant": False,
                    "tp_r": 1.5,
                    "timeout_hours": 4,
                    "sl_buffer_pips": 1.0,
                    "long_entry_buffer_pips": 0.3,
                    "short_entry_buffer_pips": 0.0,
                    "confirmation_window": "+1h_+2h",
                    "confirmation_mode": "close_reclaim",
                    "confirmation_mode_label": "Reclaim simple por close M5",
                    "confirmation_pick": "first",
                    "confirmation_pick_label": "Primera confirmacion elegible",
                    "level_profile": "all_levels",
                    "level_profile_label": "PD + Asia + London",
                    "news_mode": "sweep_plus_minus_30m",
                    "news_mode_label": "Bloqueo +/-30m alrededor del sweep",
                    "sample_size": 100,
                    "trades_per_month": 2.0,
                    "win_rate": 0.6,
                    "pf": 2.0,
                    "expectancy": 0.4,
                    "avg_R": 0.4,
                    "max_drawdown_R": -6.0,
                    "median_drawdown_R": -2.0,
                    "avg_hold_minutes": 120.0,
                    "timeout_exit_rate": 0.3,
                    "year_positive_ratio": 1.0,
                    "yearly_total_R_std": 5.0,
                    "sweeps_considered": 500,
                    "trades_executed": 100,
                    "news_blocked": 10,
                    "daily_limit_skipped": 250,
                    "no_scbi_window": 0,
                    "no_scbi_found": 20,
                    "no_entry_bar_after_scbi": 2,
                    "invalid_risk": 1,
                    "level_filtered": 0,
                    "result_by_year_json": "{}",
                    "result_by_level_type_json": "{}",
                    "result_by_level_name_json": "{}",
                    "result_by_weekday_json": "{}",
                    "result_pre_news_json": "{}",
                    "result_post_news_json": "{}",
                },
                {
                    "variant_id": "variant_a",
                    "profile_name": "axis_scan",
                    "truth_model": False,
                    "research_status": "RESEARCH_ONLY",
                    "promotion_status": "NO_PRODUCTION",
                    "experimental_variant": False,
                    "tp_r": 1.5,
                    "timeout_hours": 4,
                    "sl_buffer_pips": 1.0,
                    "long_entry_buffer_pips": 0.3,
                    "short_entry_buffer_pips": 0.0,
                    "confirmation_window": "+0h_+1h",
                    "confirmation_mode": "close_reclaim",
                    "confirmation_mode_label": "Reclaim simple por close M5",
                    "confirmation_pick": "first",
                    "confirmation_pick_label": "Primera confirmacion elegible",
                    "level_profile": "all_levels",
                    "level_profile_label": "PD + Asia + London",
                    "news_mode": "sweep_plus_minus_30m",
                    "news_mode_label": "Bloqueo +/-30m alrededor del sweep",
                    "sample_size": 110,
                    "trades_per_month": 2.2,
                    "win_rate": 0.61,
                    "pf": 2.2,
                    "expectancy": 0.45,
                    "avg_R": 0.45,
                    "max_drawdown_R": -7.0,
                    "median_drawdown_R": -2.5,
                    "avg_hold_minutes": 100.0,
                    "timeout_exit_rate": 0.2,
                    "year_positive_ratio": 1.0,
                    "yearly_total_R_std": 4.0,
                    "sweeps_considered": 510,
                    "trades_executed": 110,
                    "news_blocked": 10,
                    "daily_limit_skipped": 255,
                    "no_scbi_window": 0,
                    "no_scbi_found": 18,
                    "no_entry_bar_after_scbi": 2,
                    "invalid_risk": 1,
                    "level_filtered": 0,
                    "result_by_year_json": "{}",
                    "result_by_level_type_json": "{}",
                    "result_by_level_name_json": "{}",
                    "result_by_weekday_json": "{}",
                    "result_pre_news_json": "{}",
                    "result_post_news_json": "{}",
                },
            ]
        )
        baseline_payload = {"metrics": {"variant_id": "baseline_truth_model"}, "coverage": {}}
        files = write_matrix_outputs(self.tmp_root, ranked_results=results, profile="test", baseline_payload=baseline_payload)
        self.assertTrue(files["research_summary_json"].exists())
        self.assertTrue(files["candidate_dossier_md"].exists())
        summary = json.loads(files["research_summary_json"].read_text(encoding="utf-8"))
        self.assertEqual(summary["variant_count"], 2)

    def test_summarize_existing_results_and_cli(self) -> None:
        results_csv = self.tmp_root / "seed_results.csv"
        baseline_json = self.tmp_root / "baseline_summary.json"
        output_dir = self.tmp_root / "summary_refresh"
        pd.DataFrame(
            [
                {
                    "variant_id": "baseline_truth_model",
                    "profile_name": "baseline",
                    "truth_model": "True",
                    "research_status": "RESEARCH_ONLY",
                    "promotion_status": "NO_PRODUCTION",
                    "experimental_variant": False,
                    "tp_r": 1.5,
                    "timeout_hours": 4,
                    "sl_buffer_pips": 1.0,
                    "long_entry_buffer_pips": 0.3,
                    "short_entry_buffer_pips": 0.0,
                    "confirmation_window": "+1h_+2h",
                    "confirmation_mode": "close_reclaim",
                    "confirmation_mode_label": "Reclaim simple por close M5",
                    "confirmation_pick": "first",
                    "confirmation_pick_label": "Primera confirmacion elegible",
                    "level_profile": "all_levels",
                    "level_profile_label": "PD + Asia + London",
                    "news_mode": "sweep_plus_minus_30m",
                    "news_mode_label": "Bloqueo +/-30m alrededor del sweep",
                    "sample_size": 100,
                    "trades_per_month": 2.0,
                    "win_rate": 0.6,
                    "pf": 2.0,
                    "expectancy": 0.4,
                    "avg_R": 0.4,
                    "max_drawdown_R": -6.0,
                    "median_drawdown_R": -2.0,
                    "avg_hold_minutes": 120.0,
                    "timeout_exit_rate": 0.3,
                    "year_positive_ratio": 1.0,
                    "yearly_total_R_std": 5.0,
                    "sweeps_considered": 500,
                    "trades_executed": 100,
                    "news_blocked": 10,
                    "daily_limit_skipped": 250,
                    "no_scbi_window": 0,
                    "no_scbi_found": 20,
                    "no_entry_bar_after_scbi": 2,
                    "invalid_risk": 1,
                    "level_filtered": 0,
                    "result_by_year_json": "{}",
                    "result_by_level_type_json": "{}",
                    "result_by_level_name_json": "{}",
                    "result_by_weekday_json": "{}",
                    "result_pre_news_json": "{}",
                    "result_post_news_json": "{}",
                }
            ]
        ).to_csv(results_csv, index=False)
        baseline_json.write_text(json.dumps({"metrics": {"variant_id": "baseline_truth_model"}, "coverage": {}}, ensure_ascii=False), encoding="utf-8")
        files = summarize_existing_results(results_csv, output_dir=output_dir, profile="test", baseline_payload=json.loads(baseline_json.read_text(encoding="utf-8")))
        self.assertTrue(files["research_top_variants_csv"].exists())

        cli_output = self.tmp_root / "cli_baseline"
        command = [
            sys.executable,
            str(ROOT / "run_candidate_baseline.py"),
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-31",
            "--project-root",
            str(ROOT),
            "--output-dir",
            str(cli_output),
        ]
        completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=True)
        self.assertIn("[SUMMARY] Baseline completa", completed.stdout)
        self.assertTrue((cli_output / "baseline_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
