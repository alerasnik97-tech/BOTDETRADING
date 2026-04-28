from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from phase19_repaired_preflight import run_preflight


class M3ManifestStrictTests(unittest.TestCase):
    def test_manifest_sin_mask_cuando_required_falla(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bid = root / "bid.csv"
            ask = root / "ask.csv"
            spread = root / "spread.csv"
            news = root / "news.json"
            for path in [bid, ask, spread]:
                path.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
            news.write_text(json.dumps({"verdict": "NEWS_GUARD_STRICT_CERTIFIED"}), encoding="utf-8")
            common = {
                "certification_status": "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK",
                "source_type": "M3_FROM_M1_BID_ASK",
                "coverage_start": "2020-01-01T00:00:00+00:00",
                "coverage_end": "2026-04-01T00:00:00+00:00",
                "source": "EURUSD_M1_BID_FULL_2020_2026.csv",
                "requires_data_quality_mask": True,
                "no_interpolation": True,
                "no_forward_fill_for_trading": True,
                "synthetic_ticks": False,
            }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "period_2020_2026": {
                            "m3_bid": {"path": str(bid), **common},
                            "m3_ask": {"path": str(ask), **common},
                            "m3_spread": {"path": str(spread), **common},
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = run_preflight(manifest, news, root)
            self.assertEqual(result["verdict"], "PHASE19_REPAIRED_PREFLIGHT_BLOCKED")
            self.assertIn("M3_DATA_QUALITY_MASK_MISSING", result["blockers"])

    def test_no_m3_desde_m5(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bid = root / "bid.csv"
            ask = root / "ask.csv"
            spread = root / "spread.csv"
            mask = root / "mask.csv"
            news = root / "news.json"
            for path in [bid, ask, spread, mask]:
                path.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
            news.write_text(json.dumps({"verdict": "NEWS_GUARD_STRICT_CERTIFIED"}), encoding="utf-8")
            common = {
                "certification_status": "M3_BID_ASK_CERTIFIED_FULL",
                "source_type": "M3_FROM_M1_BID_ASK",
                "coverage_start": "2020-01-01T00:00:00+00:00",
                "coverage_end": "2026-04-01T00:00:00+00:00",
                "source": "EURUSD_M5_BID.csv",
                "requires_data_quality_mask": False,
                "no_interpolation": True,
                "no_forward_fill_for_trading": True,
                "synthetic_ticks": False,
            }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "period_2020_2026": {
                            "m3_bid": {"path": str(bid), **common},
                            "m3_ask": {"path": str(ask), **common},
                            "m3_spread": {"path": str(spread), **common},
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = run_preflight(manifest, news, root)
            self.assertIn("M3_BID_FORBIDDEN_SOURCE", result["blockers"])


if __name__ == "__main__":
    unittest.main()
