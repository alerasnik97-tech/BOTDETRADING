from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.build_am_grade_news_dataset import build_am_grade_news_dataset
from research_lab.config import NY_TZ


class AMNewsBuilderTests(unittest.TestCase):
    @contextmanager
    def _workspace_tempdir(self):
        root = Path(__file__).resolve().parent / "_tmp"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.__class__.__name__}_{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_build_am_grade_news_dataset_reports_expected_gaps(self) -> None:
        with self._workspace_tempdir() as tmp:
            output = tmp / "news_eurusd_am_fortress_v3.csv"
            summary = build_am_grade_news_dataset(output_path=output)
            self.assertTrue(output.exists())
            self.assertEqual(summary["module_verdict"], "READY_FOR_STRICT_AM_RESEARCH")
            self.assertEqual(summary["family_coverage"]["retail sales m/m"], "OFFICIAL_ANCHOR")
            self.assertEqual(summary["family_coverage"]["unemployment claims"], "OFFICIAL_ANCHOR")
            self.assertNotIn("ecb press conference", summary["critical_missing_families"])
            self.assertEqual(summary["family_coverage"]["non-farm employment change"], "OFFICIAL_ANCHOR")
            self.assertEqual(summary["family_coverage"]["ism manufacturing pmi"], "SUPPLEMENTAL_LEGACY_EXACT_PASS")
            self.assertEqual(summary["family_coverage"]["ecb press conference"], "OFFICIAL_ANCHOR")

    def test_output_keeps_nfp_at_exact_0830_ny(self) -> None:
        with self._workspace_tempdir() as tmp:
            output = tmp / "news_eurusd_am_fortress_v3.csv"
            build_am_grade_news_dataset(output_path=output)
            frame = pd.read_csv(output, dtype=str, keep_default_na=False, low_memory=False)
            nfp = frame.loc[
                (frame["event_name_normalized"] == "non-farm employment change")
                & (frame["news_source_tier"] == "official_anchor")
            ].copy()
            self.assertFalse(nfp.empty)
            ny_times = pd.to_datetime(nfp["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ).dt.strftime("%H:%M").unique().tolist()
            self.assertEqual(ny_times, ["08:30"])

    def test_output_derives_ecb_press_conference_with_dst_sensitive_ny_times(self) -> None:
        with self._workspace_tempdir() as tmp:
            output = tmp / "news_eurusd_am_fortress_v3.csv"
            build_am_grade_news_dataset(output_path=output)
            frame = pd.read_csv(output, dtype=str, keep_default_na=False, low_memory=False)
            ecb_pc = frame.loc[
                (frame["event_name_normalized"] == "ecb press conference")
                & (frame["news_source_tier"] == "official_anchor")
            ].copy()
            self.assertFalse(ecb_pc.empty)
            ny_times = sorted(
                pd.to_datetime(ecb_pc["timestamp_ny"], utc=True, errors="coerce")
                .dt.tz_convert(NY_TZ)
                .dt.strftime("%H:%M")
                .unique()
                .tolist()
            )
            self.assertEqual(ny_times, ["08:45", "09:45"])
            dst_gap_row = ecb_pc.loc[
                pd.to_datetime(ecb_pc["timestamp_ny"], utc=True, errors="coerce")
                .dt.tz_convert(NY_TZ)
                .dt.strftime("%Y-%m-%d")
                .eq("2025-10-30")
            ].copy()
            self.assertEqual(len(dst_gap_row), 1)
            dst_gap_time = pd.to_datetime(dst_gap_row.iloc[0]["timestamp_ny"], utc=True, errors="coerce").tz_convert(NY_TZ).strftime("%H:%M")
            self.assertEqual(dst_gap_time, "09:45")


if __name__ == "__main__":
    unittest.main()
