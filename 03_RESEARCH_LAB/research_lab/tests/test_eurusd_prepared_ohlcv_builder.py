from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.data_loader import OHLCV_COLUMNS, load_prepared_ohlcv
from research_lab.data_preparation.eurusd_prepared_ohlcv_builder import build_prepared_ohlcv


class EurusdPreparedOhlcvBuilderTests(unittest.TestCase):
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

    def _write_raw_month(self, raw_dir: Path, year: int, month: int, rows: list[dict[str, object]]) -> None:
        frame = pd.DataFrame(rows)
        frame.to_parquet(raw_dir / f"EURUSD_ticks_{year:04d}_{month:02d}.parquet", index=False)

    def test_dry_run_reports_targets_without_writing_files(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_dir = tmp / "raw"
            output_dir = tmp / "prepared"
            raw_dir.mkdir()
            self._write_raw_month(
                raw_dir,
                2024,
                12,
                [
                    {
                        "timestamp_utc": "2024-12-30T10:00:01Z",
                        "bid": 1.1000,
                        "ask": 1.1002,
                        "bid_volume": 1.0,
                        "ask_volume": 1.0,
                        "source": "unit",
                        "symbol": "EURUSD",
                    },
                ],
            )
            self._write_raw_month(
                raw_dir,
                2025,
                1,
                [
                    {
                        "timestamp_utc": "2025-01-02T10:00:01Z",
                        "bid": 1.2000,
                        "ask": 1.2002,
                        "bid_volume": 1.0,
                        "ask_volume": 1.0,
                        "source": "unit",
                        "symbol": "EURUSD",
                    },
                ],
            )

            summary = build_prepared_ohlcv(raw_dir=raw_dir, output_dir=output_dir, dry_run=True)

            self.assertEqual(summary["status"], "DRY_RUN_OK")
            self.assertEqual(summary["stats"]["source_files_used"], 1)
            self.assertEqual(summary["stats"]["source_files_excluded_2025_2026"], 1)
            self.assertFalse(output_dir.exists())

    def test_build_creates_loader_compatible_train_only_ohlcv(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_dir = tmp / "raw"
            output_dir = tmp / "prepared"
            raw_dir.mkdir()
            self._write_raw_month(
                raw_dir,
                2024,
                12,
                [
                    {
                        "timestamp_utc": "2024-12-30T10:00:01Z",
                        "bid": 1.1000,
                        "ask": 1.1002,
                        "bid_volume": 1.0,
                        "ask_volume": 1.0,
                        "source": "unit",
                        "symbol": "EURUSD",
                    },
                    {
                        "timestamp_utc": "2024-12-30T10:04:59Z",
                        "bid": 1.1005,
                        "ask": 1.1007,
                        "bid_volume": 1.0,
                        "ask_volume": 1.0,
                        "source": "unit",
                        "symbol": "EURUSD",
                    },
                    {
                        "timestamp_utc": "2024-12-30T10:05:00Z",
                        "bid": 1.1002,
                        "ask": 1.1004,
                        "bid_volume": 1.0,
                        "ask_volume": 1.0,
                        "source": "unit",
                        "symbol": "EURUSD",
                    },
                    {
                        "timestamp_utc": "2024-12-30T10:07:00Z",
                        "bid": 1.1008,
                        "ask": 1.1010,
                        "bid_volume": 1.0,
                        "ask_volume": 1.0,
                        "source": "unit",
                        "symbol": "EURUSD",
                    },
                ],
            )
            self._write_raw_month(
                raw_dir,
                2025,
                1,
                [
                    {
                        "timestamp_utc": "2025-01-02T10:00:01Z",
                        "bid": 1.2000,
                        "ask": 1.2002,
                        "bid_volume": 1.0,
                        "ask_volume": 1.0,
                        "source": "unit",
                        "symbol": "EURUSD",
                    },
                ],
            )

            summary = build_prepared_ohlcv(raw_dir=raw_dir, output_dir=output_dir, max_date="2024-12-31")

            self.assertEqual(summary["status"], "BUILT_OK")
            for timeframe in ("M1", "M5", "M15", "H1"):
                self.assertTrue((output_dir / f"EURUSD_{timeframe}.csv").exists())

            loaded = load_prepared_ohlcv("EURUSD", [output_dir], "M5")
            self.assertEqual(list(loaded.columns), OHLCV_COLUMNS)
            self.assertFalse(bool((loaded.index.tz_convert("UTC") >= pd.Timestamp("2025-01-01", tz="UTC")).any()))
            self.assertTrue(loaded.index.is_monotonic_increasing)
            self.assertFalse(bool(loaded.index.duplicated().any()))
            row = loaded.iloc[0]
            self.assertAlmostEqual(float(row["open"]), 1.1001)
            self.assertAlmostEqual(float(row["high"]), 1.1006)
            self.assertAlmostEqual(float(row["low"]), 1.1001)
            self.assertAlmostEqual(float(row["close"]), 1.1003)
            self.assertAlmostEqual(float(row["volume"]), 3.0)

    def test_missing_news_file_does_not_block_core_loader_when_news_disabled(self) -> None:
        with self._workspace_tempdir() as tmp:
            data_dir = tmp / "prepared"
            data_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "timestamp": "2024-12-30T10:05:00+00:00",
                        "open": 1.1000,
                        "high": 1.1005,
                        "low": 1.0998,
                        "close": 1.1002,
                        "volume": 3.0,
                    }
                ]
            ).set_index("timestamp").to_csv(data_dir / "EURUSD_M5.csv")
            loaded = load_prepared_ohlcv("EURUSD", [data_dir], "M5")
            self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
