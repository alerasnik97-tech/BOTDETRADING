from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from phase19_repaired_preflight import run_preflight


class Phase19RepairedRequiresM3Tests(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_manifest_without_m3_blocks_phase19_repaired(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            news = root / "news_audit.json"
            self.write_json(manifest, {"period_2020_2026": {}})
            self.write_json(news, {"verdict": "NEWS_GUARD_STRICT_CERTIFIED"})
            result = run_preflight(manifest, news, root)
            self.assertEqual(result["verdict"], "PHASE19_REPAIRED_PREFLIGHT_BLOCKED")
            self.assertIn("M3_BID_MISSING", result["blockers"])

    def test_manifest_with_certified_m3_unlocks_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bid = root / "m3_bid.csv"
            ask = root / "m3_ask.csv"
            spread = root / "m3_spread.csv"
            for p in [bid, ask, spread]:
                p.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
            common = {
                "rows": 100,
                "start": "2020-01-01T00:00:00+00:00",
                "end": "2026-04-01T00:00:00+00:00",
                "timezone": "UTC",
                "certification_status": "M3_BID_ASK_CERTIFIED",
                "source_type": "m1_derived",
                "source": str(root / "EURUSD_M1_BID_FULL_2020_2026.csv"),
            }
            manifest = root / "manifest.json"
            news = root / "news_audit.json"
            self.write_json(
                manifest,
                {
                    "period_2020_2026": {
                        "m3_bid": {"path": str(bid), **common},
                        "m3_ask": {"path": str(ask), **common},
                        "m3_spread": {"path": str(spread), **common},
                    }
                },
            )
            self.write_json(news, {"verdict": "NEWS_GUARD_STRICT_CERTIFIED"})
            result = run_preflight(manifest, news, root)
            self.assertEqual(result["verdict"], "PHASE19_REPAIRED_PREFLIGHT_PASSED")


if __name__ == "__main__":
    unittest.main()
