from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.price_source_audit import build_price_source_recommendation, discover_price_sources


class PriceSourceAuditTests(unittest.TestCase):
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

    def _write_prepared(self, path: Path, start: str = "2025-01-02 11:00:00-05:00") -> None:
        frame = pd.DataFrame(
            {
                "timestamp": [
                    start,
                    "2025-01-02 11:05:00-05:00",
                ],
                "open": [1.1, 1.2],
                "high": [1.2, 1.3],
                "low": [1.0, 1.1],
                "close": [1.15, 1.25],
                "volume": [10, 12],
            }
        ).set_index("timestamp")
        frame.to_csv(path)

    def test_discover_price_sources_prefers_bid_ask_m1_over_m5(self) -> None:
        with self._workspace_tempdir() as tmp:
            self._write_prepared(tmp / "EURUSD_M5.csv")
            self._write_prepared(tmp / "EURUSD_M1_BID.csv")
            self._write_prepared(tmp / "EURUSD_M1_ASK.csv")
            discovered = discover_price_sources("EURUSD", [tmp])
            self.assertGreaterEqual(len(discovered), 3)
            self.assertEqual(discovered[0].timeframe, "M1")
            self.assertEqual(discovered[0].side_coverage, "bid_ask")

    def test_build_recommendation_flags_missing_tick_and_bid_ask(self) -> None:
        with self._workspace_tempdir() as tmp:
            self._write_prepared(tmp / "EURUSD_M5.csv")
            payload = build_price_source_recommendation("EURUSD", [tmp])
            self.assertIn("no_hay_BID_ASK_historico_local; el ASK sigue siendo sintetico", payload["current_limitations"])
            self.assertIn("no_hay_tick_data_local; la politica intrabar sigue dependiendo de OHLC", payload["current_limitations"])
