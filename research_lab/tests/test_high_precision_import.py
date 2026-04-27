from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.data_loader import load_high_precision_package
from research_lab.high_precision_import import expected_input_schema, integrate_high_precision_source


class HighPrecisionImportTests(unittest.TestCase):
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

    def _write_ohlcv(self, path: Path, rows: list[tuple[str, float, float, float, float, float]]) -> None:
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
        frame.to_csv(path)

    def _write_ticks(self, path: Path, rows: list[tuple[str, float, float]]) -> None:
        frame = pd.DataFrame(rows, columns=["timestamp", "price", "volume"]).set_index("timestamp")
        frame.to_csv(path)

    def test_expected_schema_describes_required_files(self) -> None:
        schema = expected_input_schema("dukascopy_m1_bid_ask")
        self.assertEqual(schema["expected_files"], ["EURUSD_M1_BID.csv", "EURUSD_M1_ASK.csv"])

    def test_integrate_m1_bid_ask_writes_manifest_and_mid(self) -> None:
        with self._workspace_tempdir() as tmp:
            input_dir = tmp / "raw"
            output_dir = tmp / "prepared"
            input_dir.mkdir()
            self._write_ohlcv(
                input_dir / "EURUSD_M1_BID.csv",
                [
                    ("2025-01-02 11:00:00-05:00", 1.1000, 1.1005, 1.0995, 1.1002, 10),
                    ("2025-01-02 11:01:00-05:00", 1.1002, 1.1007, 1.0998, 1.1004, 12),
                ],
            )
            self._write_ohlcv(
                input_dir / "EURUSD_M1_ASK.csv",
                [
                    ("2025-01-02 11:00:00-05:00", 1.1002, 1.1007, 1.0997, 1.1004, 10),
                    ("2025-01-02 11:01:00-05:00", 1.1004, 1.1009, 1.1000, 1.1006, 12),
                ],
            )
            manifest = integrate_high_precision_source(
                pair="EURUSD",
                source_type="dukascopy_m1_bid_ask",
                input_dir=input_dir,
                output_dir=output_dir,
            )
            self.assertTrue((output_dir / "EURUSD_M1_BID.csv").exists())
            self.assertTrue((output_dir / "EURUSD_M1_ASK.csv").exists())
            self.assertTrue((output_dir / "EURUSD_M1_MID.csv").exists())
            self.assertTrue((output_dir / "EURUSD_high_precision_manifest.json").exists())
            mid = pd.read_csv(output_dir / "EURUSD_M1_MID.csv", index_col=0)
            self.assertAlmostEqual(float(mid.iloc[0]["open"]), 1.1001, places=7)
            self.assertEqual(manifest["rows_mid"], 2)

    def test_integrate_tick_bid_ask_resamples_to_m1(self) -> None:
        with self._workspace_tempdir() as tmp:
            input_dir = tmp / "raw"
            output_dir = tmp / "prepared"
            input_dir.mkdir()
            self._write_ticks(
                input_dir / "EURUSD_TICK_BID.csv",
                [
                    ("2025-01-02 11:00:10-05:00", 1.1000, 1),
                    ("2025-01-02 11:00:40-05:00", 1.1003, 1),
                    ("2025-01-02 11:01:10-05:00", 1.1002, 1),
                ],
            )
            self._write_ticks(
                input_dir / "EURUSD_TICK_ASK.csv",
                [
                    ("2025-01-02 11:00:10-05:00", 1.1002, 1),
                    ("2025-01-02 11:00:40-05:00", 1.1005, 1),
                    ("2025-01-02 11:01:10-05:00", 1.1004, 1),
                ],
            )
            manifest = integrate_high_precision_source(
                pair="EURUSD",
                source_type="dukascopy_tick_bid_ask",
                input_dir=input_dir,
                output_dir=output_dir,
            )
            bid = pd.read_csv(output_dir / "EURUSD_M1_BID.csv", index_col=0)
            ask = pd.read_csv(output_dir / "EURUSD_M1_ASK.csv", index_col=0)
            self.assertGreaterEqual(len(bid), 2)
            self.assertEqual(len(bid), len(ask))
            loaded_manifest = json.loads((output_dir / "EURUSD_high_precision_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded_manifest["source_type"], "dukascopy_tick_bid_ask")
            self.assertEqual(manifest["source_type"], "dukascopy_tick_bid_ask")

    def test_load_high_precision_package_reads_generated_m1_files(self) -> None:
        with self._workspace_tempdir() as tmp:
            input_dir = tmp / "raw"
            output_dir = tmp / "prepared"
            input_dir.mkdir()
            self._write_ohlcv(
                input_dir / "EURUSD_M1_BID.csv",
                [
                    ("2025-01-02 11:00:00-05:00", 1.1000, 1.1005, 1.0995, 1.1002, 10),
                    ("2025-01-02 11:01:00-05:00", 1.1002, 1.1007, 1.0998, 1.1004, 12),
                ],
            )
            self._write_ohlcv(
                input_dir / "EURUSD_M1_ASK.csv",
                [
                    ("2025-01-02 11:00:00-05:00", 1.1002, 1.1007, 1.0997, 1.1004, 10),
                    ("2025-01-02 11:01:00-05:00", 1.1004, 1.1009, 1.1000, 1.1006, 12),
                ],
            )
            integrate_high_precision_source(
                pair="EURUSD",
                source_type="dukascopy_m1_bid_ask",
                input_dir=input_dir,
                output_dir=output_dir,
            )
            package = load_high_precision_package("EURUSD", output_dir)
            self.assertEqual(sorted(package.keys()), ["ask", "bid", "mid"])
            self.assertEqual(len(package["mid"]), 2)
