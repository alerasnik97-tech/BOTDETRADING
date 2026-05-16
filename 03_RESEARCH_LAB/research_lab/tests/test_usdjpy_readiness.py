from __future__ import annotations

import unittest
from pathlib import Path

from research_lab.build_usdjpy_news_fortress_dataset import DEFAULT_JPY_RAW_FILE, build_usdjpy_news_fortress_dataset
from research_lab.config import (
    canonical_news_config,
    canonical_news_file,
    canonical_prepared_data_dirs,
    first_family_requires_high_precision,
)
from research_lab.engine import quote_to_usd
from research_lab.news_filter import load_news_events


class USDJPYReadinessTests(unittest.TestCase):
    def _skip_if_missing_jpy_raw_news(self) -> None:
        if not Path(DEFAULT_JPY_RAW_FILE).exists():
            self.skipTest(f"SKIPPED_MISSING_REQUIRED_DATA: {DEFAULT_JPY_RAW_FILE}")

    def test_usdjpy_canonical_data_dirs_are_pair_specific_and_present(self) -> None:
        data_dirs = canonical_prepared_data_dirs("USDJPY")
        self.assertEqual(
            data_dirs,
            (
                Path("05_MARKET_DATA_VAULT/legacy_data/data_usdjpy_2016_2021/prepared"),
                Path("05_MARKET_DATA_VAULT/legacy_data/data_usdjpy_2022_2025/prepared"),
            ),
        )
        for data_dir in data_dirs:
            self.assertTrue((data_dir / "USDJPY_M5.csv").exists())
            self.assertTrue((data_dir / "USDJPY_M15.csv").exists())
            self.assertTrue((data_dir / "USDJPY_H1.csv").exists())

    def test_usdjpy_quote_to_usd_uses_inverse_price(self) -> None:
        self.assertAlmostEqual(quote_to_usd("USDJPY", 150.0), 1.0 / 150.0, places=12)

    def test_usdjpy_first_family_does_not_require_high_precision(self) -> None:
        self.assertFalse(first_family_requires_high_precision("USDJPY"))

    def test_build_usdjpy_news_fortress_dataset_closes_critical_families(self) -> None:
        self._skip_if_missing_jpy_raw_news()
        summary = build_usdjpy_news_fortress_dataset()
        self.assertTrue(Path(summary["clean_dataset_path"]).exists())
        self.assertEqual(summary["module_verdict"], "USDJPY_READY_FOR_FIRST_RESEARCH_FAMILY_DESIGN")
        self.assertEqual(summary["critical_missing_families"], [])
        self.assertGreater(summary["usd_rows_reused"], 0)
        self.assertGreater(summary["jpy_rows_curated"], 0)
        self.assertEqual(summary["family_coverage"]["boj policy rate"], "JPY_CURATED_LOCAL_EXACT")
        self.assertEqual(summary["family_coverage"]["monetary policy statement"], "JPY_CURATED_LOCAL_EXACT")
        self.assertEqual(summary["family_coverage"]["boj press conference"], "JPY_CURATED_LOCAL_EXACT")

    def test_usdjpy_canonical_news_config_loads_usd_and_jpy_events(self) -> None:
        self._skip_if_missing_jpy_raw_news()
        build_usdjpy_news_fortress_dataset()
        settings = canonical_news_config("USDJPY")
        self.assertEqual(canonical_news_file("USDJPY"), settings.file_path)
        result = load_news_events("USDJPY", settings)
        self.assertTrue(result.enabled)
        currencies = set(result.events["currency"].astype(str).str.upper().tolist())
        self.assertIn("USD", currencies)
        self.assertIn("JPY", currencies)


if __name__ == "__main__":
    unittest.main()
