from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.config import DEFAULT_DATA_DIRS, PAIR_CANONICAL_DATA_DIRS
from research_lab.data_preparation.eurusd_prepared_ohlcv_builder import build_prepared_ohlcv, write_holdout_seal


class EurusdHoldoutSealTests(unittest.TestCase):
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

    def test_default_data_dirs_do_not_include_sealed_holdout(self) -> None:
        default_paths = [str(path).replace("\\", "/") for path in DEFAULT_DATA_DIRS]
        canonical_paths = [str(path).replace("\\", "/") for path in PAIR_CANONICAL_DATA_DIRS["EURUSD"]]
        for value in default_paths + canonical_paths:
            self.assertNotIn("sealed_holdout_2025_2026", value)
            self.assertNotIn("2025_2026", value)

    def test_train_and_holdout_partitions_are_disjoint(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_dir = tmp / "raw"
            train_dir = tmp / "train"
            holdout_dir = tmp / "holdout"
            raw_dir.mkdir()
            self._write_raw_month(
                raw_dir,
                2024,
                12,
                [
                    {"timestamp_utc": "2024-12-31T21:59:00Z", "bid": 1.1000, "ask": 1.1002, "symbol": "EURUSD"},
                    {"timestamp_utc": "2024-12-31T21:59:30Z", "bid": 1.1001, "ask": 1.1003, "symbol": "EURUSD"},
                ],
            )
            self._write_raw_month(
                raw_dir,
                2025,
                1,
                [
                    {"timestamp_utc": "2025-01-01T22:00:00Z", "bid": 1.2000, "ask": 1.2002, "symbol": "EURUSD"},
                    {"timestamp_utc": "2025-01-01T22:00:30Z", "bid": 1.2001, "ask": 1.2003, "symbol": "EURUSD"},
                ],
            )

            train = build_prepared_ohlcv(raw_dir=raw_dir, output_dir=train_dir, partition="train", max_date="2024-12-31")
            holdout = build_prepared_ohlcv(raw_dir=raw_dir, output_dir=holdout_dir, partition="holdout", min_date="2025-01-01")

            self.assertLess(pd.Timestamp(train["timeframes"]["M1"]["max_timestamp_utc"]), pd.Timestamp("2025-01-01T00:00:00Z"))
            self.assertGreaterEqual(pd.Timestamp(holdout["timeframes"]["M1"]["min_timestamp_utc"]), pd.Timestamp("2025-01-01T00:00:00Z"))
            self.assertEqual(holdout["safety"]["sealed_not_for_research_selection"], True)

    def test_holdout_seal_manifest_is_written_with_no_default_loader_access(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_dir = tmp / "raw"
            holdout_dir = tmp / "holdout"
            manifest_dir = tmp / "manifests"
            raw_dir.mkdir()
            self._write_raw_month(
                raw_dir,
                2025,
                1,
                [
                    {"timestamp_utc": "2025-01-01T22:00:00Z", "bid": 1.2000, "ask": 1.2002, "symbol": "EURUSD"},
                    {"timestamp_utc": "2025-01-01T22:00:30Z", "bid": 1.2001, "ask": 1.2003, "symbol": "EURUSD"},
                ],
            )
            summary = build_prepared_ohlcv(raw_dir=raw_dir, output_dir=holdout_dir, partition="holdout", min_date="2025-01-01")
            write_holdout_seal(summary, manifest_dir)

            manifest = pd.read_csv(manifest_dir / "EURUSD_HOLDOUT_MANIFEST.csv")
            self.assertEqual(set(manifest["seal_status"]), {"SEALED_NOT_FOR_RESEARCH_SELECTION"})
            self.assertEqual(set(manifest["default_loader_access"]), {"NO"})
            self.assertTrue((manifest_dir / "EURUSD_HOLDOUT_SEAL_REPORT.json").exists())


if __name__ == "__main__":
    unittest.main()
