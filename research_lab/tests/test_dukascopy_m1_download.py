from __future__ import annotations

import unittest
from datetime import date

from research_lab.dukascopy_m1_download import _dukascopy_instrument, _month_chunks


class DukascopyM1DownloadTests(unittest.TestCase):
    def test_dukascopy_instrument_formats_eurusd(self) -> None:
        self.assertEqual(_dukascopy_instrument("EURUSD"), "EUR/USD")

    def test_month_chunks_split_month_boundaries(self) -> None:
        chunks = _month_chunks(date(2024, 10, 1), date(2024, 12, 5))
        self.assertEqual(
            [(chunk.start.isoformat(), chunk.end_inclusive.isoformat()) for chunk in chunks],
            [
                ("2024-10-01", "2024-10-31"),
                ("2024-11-01", "2024-11-30"),
                ("2024-12-01", "2024-12-05"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
