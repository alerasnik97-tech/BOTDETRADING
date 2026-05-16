import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from research_lab.lab_preflight import (
    assert_train_data_no_holdout,
    assert_default_data_dirs_no_holdout,
    assert_output_dir_not_quarantine,
    assert_news_disabled_unless_certified
)

class TestLabPreflight(unittest.TestCase):
    def test_assert_train_data_no_holdout_passes_clean_data(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame(index=idx)
        try:
            assert_train_data_no_holdout(df)
        except RuntimeError:
            self.fail("assert_train_data_no_holdout raised RuntimeError unexpectedly")

    def test_assert_train_data_no_holdout_fails_leaky_data(self):
        idx = pd.date_range("2024-12-30", periods=10, freq="D", tz="UTC")
        # Starts 2024-12-30, goes into Jan 2025
        df = pd.DataFrame(index=idx)
        with self.assertRaisesRegex(RuntimeError, "FAIL_CLOSED: Detected .* rows with timestamp >= 2025-01-01"):
            assert_train_data_no_holdout(df)

    def test_assert_default_data_dirs_no_holdout(self):
        # This depends on config.py, which we expect to be clean
        try:
            assert_default_data_dirs_no_holdout()
        except RuntimeError as e:
            self.fail(f"DEFAULT_DATA_DIRS check failed: {e}")

    def test_assert_output_dir_not_quarantine_fails_bad_path(self):
        bad_path = "07_BACKUPS/quarantine/results"
        with self.assertRaises(RuntimeError):
            assert_output_dir_not_quarantine(bad_path)

    def test_assert_output_dir_not_quarantine_passes_good_path(self):
        good_path = "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_rerun"
        try:
            assert_output_dir_not_quarantine(good_path)
        except RuntimeError:
            self.fail("assert_output_dir_not_quarantine failed on valid path")

    def test_assert_news_disabled(self):
        # DEFAULT_NEWS_ENABLED should be False in config.py
        try:
            assert_news_disabled_unless_certified()
        except RuntimeError:
            self.fail("News check failed, but should be disabled in config")

if __name__ == "__main__":
    unittest.main()
