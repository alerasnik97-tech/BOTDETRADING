from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from research_lab.config import DEFAULT_DATA_DIRS, DEFAULT_NEWS_FILE, DEFAULT_NEWS_SUMMARY_FILE, DEFAULT_RAW_NEWS_FILE, NewsConfig
from research_lab.data_loader import load_backtest_data_bundle, load_price_data
from research_lab.news_filter import filter_event_family, load_news_events, load_news_summary


class RealProjectIntegrationTests(unittest.TestCase):
    def test_default_news_config_is_disabled(self) -> None:
        self.assertFalse(NewsConfig().enabled)

    def test_real_project_data_keeps_sunday_reopen(self) -> None:
        frame = load_price_data("EURUSD", list(DEFAULT_DATA_DIRS), "2022-01-01", "2022-01-04")
        sunday_mask = (frame.index.dayofweek == 6) & ((frame.index.hour * 60 + frame.index.minute) > 17 * 60)
        self.assertGreater(int(sunday_mask.sum()), 0)

    def test_high_precision_bundle_uses_dukascopy_bid_ask_when_requested(self) -> None:
        bundle = load_backtest_data_bundle("EURUSD", list(DEFAULT_DATA_DIRS), "2024-10-01", "2024-10-07", "high_precision_mode")
        self.assertEqual(bundle.data_source_used, "dukascopy_m1_bid_ask_full")
        self.assertIsNotNone(bundle.precision_package)
        self.assertTrue(bundle.precision_package["bid_m15"].index.equals(bundle.frame.index))
        self.assertTrue(bundle.precision_package["ask_m15"].index.equals(bundle.frame.index))

    def test_current_news_source_is_disabled_until_approved(self) -> None:
        result = load_news_events(
            "EURUSD",
            NewsConfig(
                enabled=True,
                file_path=Path(DEFAULT_NEWS_FILE),
                raw_file_path=Path(DEFAULT_RAW_NEWS_FILE),
                source_approved=False,
                pre_minutes=15,
                post_minutes=15,
                currencies=("USD", "EUR"),
            ),
        )
        self.assertFalse(result.enabled)
        self.assertEqual(result.disabled_reason, "source_not_approved")
        self.assertGreater(result.approved_rows, 0)

    def test_current_validated_news_dataset_is_rejected_even_when_force_approved(self) -> None:
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
        self.assertFalse(result.enabled)
        self.assertEqual(result.disabled_reason, "source_not_approved")

    def test_news_summary_marks_operational_dataset_rejected(self) -> None:
        summary = load_news_summary(Path(DEFAULT_NEWS_SUMMARY_FILE))
        self.assertEqual(summary.get("module_verdict"), "REJECTED_DISABLED")
        self.assertFalse(bool(summary.get("source_approved")))

    def test_gdp_qq_alias_family_has_high_impact_usd_coverage(self) -> None:
        audit = pd.read_csv(Path("data/news_eurusd_m15_audit.csv"), dtype=str, keep_default_na=False)
        family = filter_event_family(audit, "gdp q/q")
        family = family.loc[family["currency"].isin(["USD", "EUR"]) & family["impact_level"].eq("HIGH")]
        self.assertGreater(len(family), 0)
        self.assertTrue(family["event_name_normalized"].isin(["advance gdp q/q", "prelim gdp q/q", "final gdp q/q"]).any())

    def test_ppi_yy_alias_family_is_absent_in_source_and_not_fabricated(self) -> None:
        audit = pd.read_csv(Path("data/news_eurusd_m15_audit.csv"), dtype=str, keep_default_na=False)
        family = filter_event_family(audit, "ppi y/y")
        family = family.loc[family["currency"].isin(["USD", "EUR"]) & family["impact_level"].eq("HIGH")]
        self.assertEqual(len(family), 0)
