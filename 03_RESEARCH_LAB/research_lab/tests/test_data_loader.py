from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.data_loader import (
    _resample_to_m15,
    describe_available_price_data,
    fx_market_mask,
    fx_session_date,
    load_prepared_ohlcv,
    load_price_data,
    parse_prepared_index,
)


class DataLoaderTests(unittest.TestCase):
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

    def _write_csv(self, directory: Path, rows: list[dict[str, object]]) -> None:
        frame = pd.DataFrame(rows).set_index("timestamp")
        frame.to_csv(directory / "EURUSD_M5.csv")

    def test_load_price_data_preserves_sunday_reopen(self) -> None:
        with self._workspace_tempdir() as tmp:
            data_dir = Path(tmp)
            self._write_csv(
                data_dir,
                [
                    {"timestamp": "2022-01-02 17:05:00-05:00", "open": 1.13, "high": 1.14, "low": 1.12, "close": 1.135, "volume": 1.0},
                    {"timestamp": "2022-01-02 17:10:00-05:00", "open": 1.135, "high": 1.14, "low": 1.13, "close": 1.138, "volume": 1.0},
                    {"timestamp": "2022-01-03 00:00:00-05:00", "open": 1.138, "high": 1.14, "low": 1.137, "close": 1.139, "volume": 1.0},
                ],
            )
            loaded = load_price_data("EURUSD", [data_dir], "2022-01-01", "2022-01-03")
            sunday_mask = (loaded.index.dayofweek == 6) & ((loaded.index.hour * 60 + loaded.index.minute) > 17 * 60)
            self.assertEqual(int(sunday_mask.sum()), 2)

    def test_load_price_data_rejects_timezone_naive_index(self) -> None:
        with self._workspace_tempdir() as tmp:
            data_dir = Path(tmp)
            self._write_csv(
                data_dir,
                [
                    {"timestamp": "2022-01-03 00:00:00", "open": 1.13, "high": 1.14, "low": 1.12, "close": 1.135, "volume": 1.0},
                ],
            )
            with self.assertRaises(ValueError):
                load_price_data("EURUSD", [data_dir], "2022-01-01", "2022-01-03")

    def test_fx_market_mask_keeps_fx_week(self) -> None:
        index = parse_prepared_index(
            pd.Index(
                [
                    "2022-01-07 17:00:00-05:00",
                    "2022-01-07 17:05:00-05:00",
                    "2022-01-08 12:00:00-05:00",
                    "2022-01-09 17:00:00-05:00",
                    "2022-01-09 17:05:00-05:00",
                ]
            )
        )
        mask = fx_market_mask(index)
        self.assertEqual(mask.tolist(), [True, False, False, False, True])

    def test_fx_session_date_maps_sunday_reopen_to_monday(self) -> None:
        index = parse_prepared_index(
            pd.Index(
                [
                    "2022-01-02 17:05:00-05:00",
                    "2022-01-02 18:00:00-05:00",
                    "2022-01-03 09:00:00-05:00",
                ]
            )
        )
        session_dates = fx_session_date(index)
        self.assertEqual(str(session_dates.iloc[0]), "2022-01-03")
        self.assertEqual(str(session_dates.iloc[1]), "2022-01-03")
        self.assertEqual(str(session_dates.iloc[2]), "2022-01-03")

    def test_resample_to_m15_builds_expected_ohlc(self) -> None:
        index = parse_prepared_index(
            pd.Index(
                [
                    "2022-01-03 11:05:00-05:00",
                    "2022-01-03 11:10:00-05:00",
                    "2022-01-03 11:15:00-05:00",
                ]
            )
        )
        frame = pd.DataFrame(
            {
                "open": [1.1000, 1.1005, 1.1008],
                "high": [1.1010, 1.1015, 1.1012],
                "low": [1.0995, 1.1001, 1.1004],
                "close": [1.1006, 1.1011, 1.1009],
                "volume": [10.0, 20.0, 30.0],
            },
            index=index,
        )
        m15 = _resample_to_m15(frame)
        self.assertEqual(len(m15), 1)
        row = m15.iloc[0]
        self.assertAlmostEqual(float(row["open"]), 1.1000)
        self.assertAlmostEqual(float(row["high"]), 1.1015)
        self.assertAlmostEqual(float(row["low"]), 1.0995)
        self.assertAlmostEqual(float(row["close"]), 1.1009)
        self.assertAlmostEqual(float(row["volume"]), 60.0)

    def test_load_prepared_ohlcv_supports_direct_timeframe_loading(self) -> None:
        with self._workspace_tempdir() as tmp:
            data_dir = Path(tmp)
            frame = pd.DataFrame(
                [
                    {"timestamp": "2022-01-03 11:15:00-05:00", "open": 1.1000, "high": 1.1015, "low": 1.0995, "close": 1.1009, "volume": 60.0},
                ]
            ).set_index("timestamp")
            frame.to_csv(data_dir / "EURUSD_M15.csv")
            loaded = load_prepared_ohlcv("EURUSD", [data_dir], "M15")
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(float(loaded.iloc[0]["close"]), 1.1009)

    def test_describe_available_price_data_reports_existing_timeframes(self) -> None:
        with self._workspace_tempdir() as tmp:
            data_dir = Path(tmp)
            self._write_csv(
                data_dir,
                [
                    {"timestamp": "2022-01-03 11:00:00-05:00", "open": 1.13, "high": 1.14, "low": 1.12, "close": 1.135, "volume": 1.0},
                ],
            )
            pd.DataFrame(
                [
                    {"timestamp": "2022-01-03 11:15:00-05:00", "open": 1.13, "high": 1.14, "low": 1.12, "close": 1.135, "volume": 1.0},
                ]
            ).set_index("timestamp").to_csv(data_dir / "EURUSD_M15.csv")
            catalog = describe_available_price_data("EURUSD", [data_dir])
            timeframes = {row["timeframe"] for row in catalog}
            self.assertEqual(timeframes, {"M5", "M15"})
