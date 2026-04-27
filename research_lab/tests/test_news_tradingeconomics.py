from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.config import NY_TZ, NewsConfig
from research_lab.news_tradingeconomics import CANONICAL_COLUMNS_V2, import_tradingeconomics_calendar


class TradingEconomicsImportTests(unittest.TestCase):
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

    def _write_json(self, path: Path, payload: list[dict[str, object]]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_import_tradingeconomics_converts_utc_to_new_york(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_calendar.json"
            self._write_json(
                raw_path,
                [
                    {
                        "CalendarId": "1001",
                        "Date": "2025-01-10T13:30:00",
                        "Country": "United States",
                        "Category": "Non Farm Payrolls",
                        "Event": "Non Farm Payrolls",
                        "Importance": "3",
                        "Source": "U.S. Bureau of Labor Statistics",
                        "SourceURL": "https://www.bls.gov/",
                    }
                ],
            )
            settings = NewsConfig(source_approved=True)
            result = import_tradingeconomics_calendar(
                raw_path,
                clean_output_path=tmp / "news_te_validated.csv",
                settings=settings,
            )
            self.assertEqual(len(result.clean_frame), 1)
            row = result.clean_frame.iloc[0]
            self.assertEqual(row["title"], "non-farm employment change")
            ts_ny = pd.Timestamp(row["scheduled_at_ny"])
            self.assertEqual(ts_ny.tz_convert(NY_TZ).strftime("%Y-%m-%d %H:%M"), "2025-01-10 08:30")
            self.assertTrue(settings.source_approved)
            self.assertTrue(bool(row["operational_eligible"]))
            self.assertEqual(result.summary.get("operational_eligible_rows"), 1)
            self.assertEqual(result.summary.get("technical_approved_rows"), 1)

    def test_import_tradingeconomics_maps_gdp_alias_family(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_calendar.json"
            self._write_json(
                raw_path,
                [
                    {
                        "CalendarId": "2001",
                        "Date": "2025-04-30T12:30:00",
                        "Country": "United States",
                        "Category": "GDP Growth Rate QoQ Adv",
                        "Event": "GDP Growth Rate QoQ Adv",
                        "Importance": "3",
                        "Source": "U.S. Bureau of Economic Analysis",
                        "SourceURL": "https://www.bea.gov/",
                    }
                ],
            )
            settings = NewsConfig(source_approved=True)
            result = import_tradingeconomics_calendar(
                raw_path,
                clean_output_path=tmp / "news_te_validated.csv",
                settings=settings,
            )
            self.assertEqual(result.clean_frame.iloc[0]["title"], "advance gdp q/q")

    def test_import_writes_schema_v2_and_operational_gate_in_summary(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_calendar.json"
            self._write_json(
                raw_path,
                [
                    {
                        "CalendarId": "3001",
                        "Date": "2025-01-10T13:30:00",
                        "Country": "United States",
                        "Category": "Unemployment Rate",
                        "Event": "Unemployment Rate",
                        "Importance": "3",
                        "Source": "U.S. Bureau of Labor Statistics",
                        "SourceURL": "https://www.bls.gov/",
                    }
                ],
            )
            clean_path = tmp / "news_te_validated.csv"
            import_tradingeconomics_calendar(
                raw_path,
                clean_output_path=clean_path,
                settings=NewsConfig(source_approved=True),
            )
            written = pd.read_csv(clean_path, dtype=str, keep_default_na=False, low_memory=False)
            for col in CANONICAL_COLUMNS_V2:
                self.assertIn(col, written.columns)
            self.assertIn("status_column_semantics", Path(tmp / "news_te_validated_summary.json").read_text(encoding="utf-8"))
