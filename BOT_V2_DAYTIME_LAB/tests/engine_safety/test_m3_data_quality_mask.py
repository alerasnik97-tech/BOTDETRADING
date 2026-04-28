from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from data_certification.m3_gap_repair import build_data_quality_mask


class M3DataQualityMaskTests(unittest.TestCase):
    def test_mask_bloquea_dia_con_gap_critico(self):
        classified = pd.DataFrame(
            [
                {
                    "start_ny": "2026-01-06T10:00:00-05:00",
                    "severity": "CRITICAL_MASK_DAY",
                    "classification": "INTRADAY_CRITICAL_PHASE19",
                    "in_phase18_window_08_11": True,
                    "in_phase19_window_08_1630": True,
                    "in_user_window_07_20": True,
                }
            ]
        )
        mask = build_data_quality_mask(classified, "2026-01-06T00:00:00+00:00", "2026-01-06T23:59:00+00:00")
        day = mask[mask["date_ny"] == "2026-01-06"].iloc[0]
        self.assertFalse(bool(day["allow_phase19_repaired"]))
        self.assertFalse(bool(day["allow_phase18"]))

    def test_gap_fuera_phase19_no_bloquea_phase19(self):
        classified = pd.DataFrame(
            [
                {
                    "start_ny": "2026-01-06T17:30:00-05:00",
                    "severity": "WARNING_MASK_SESSION",
                    "classification": "ROLLOVER_MAINTENANCE",
                    "in_phase18_window_08_11": False,
                    "in_phase19_window_08_1630": False,
                    "in_user_window_07_20": True,
                }
            ]
        )
        mask = build_data_quality_mask(classified, "2026-01-06T00:00:00+00:00", "2026-01-06T23:59:00+00:00")
        day = mask[mask["date_ny"] == "2026-01-06"].iloc[0]
        self.assertTrue(bool(day["allow_phase19_repaired"]))


if __name__ == "__main__":
    unittest.main()
