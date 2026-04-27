from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path

from research_lab.official_anchors.builder import build_canonical_dataframe
from research_lab.official_anchors.connectors.bls_employment import fetch_bls_employment_situation_events
from research_lab.official_anchors.connectors.manifest_json import fetch_from_user_manifest
from research_lab.official_anchors.schema import IntermediateEvent
from research_lab.official_anchors.time_rules import ny_local_to_utc_iso


class OfficialAnchorsPipelineTests(unittest.TestCase):
    def test_ny_to_utc_winter_vs_summer(self) -> None:
        utc_w, ny_w, _ = ny_local_to_utc_iso(date(2025, 1, 3), "08:30")
        utc_s, ny_s, _ = ny_local_to_utc_iso(date(2025, 7, 4), "08:30")
        self.assertTrue(utc_w.endswith("+00:00") or "+00:00" in utc_w)
        self.assertIn("08:30:00", ny_w)
        self.assertIn("08:30:00", ny_s)
        self.assertNotEqual(utc_w[:16], utc_s[:16])

    def test_bls_first_friday_count_2024(self) -> None:
        r = fetch_bls_employment_situation_events(date(2024, 1, 1), date(2024, 12, 31))
        self.assertEqual(r.status, "partial")
        self.assertEqual(len(r.events), 24)
        titles = {e.title for e in r.events}
        self.assertIn("non-farm employment change", titles)
        self.assertIn("unemployment rate", titles)

    def test_builder_rejects_bad_date(self) -> None:
        bad = IntermediateEvent(
            title="cpi m/m",
            country="US",
            currency="USD",
            local_date_ny="not-a-date",
            local_time_ny="08:30",
            source="test",
            source_type="test",
            source_url="",
            anchor_group="CPI",
        )
        clean, audit, stats = build_canonical_dataframe([bad], source_approved=False)
        self.assertTrue(clean.empty)
        self.assertEqual(stats["technical_approved"], 0)

    def test_manifest_json_roundtrip(self) -> None:
        tmp = Path(__file__).parent / "_tmp_manifest_anchor.json"
        tmp.write_text(
            json.dumps(
                {
                    "releases": [
                        {
                            "title": "retail sales m/m",
                            "local_date_ny": "2025-06-15",
                            "local_time_ny": "08:30",
                            "currency": "USD",
                            "country": "United States",
                            "source": "test_manifest",
                            "source_url": "https://example.invalid",
                            "anchor_group": "RETAIL",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        try:
            r = fetch_from_user_manifest(tmp)
            self.assertEqual(len(r.events), 1)
            clean, audit, stats = build_canonical_dataframe(r.events, source_approved=False)
            self.assertEqual(stats["technical_approved"], 1)
            self.assertFalse(clean.empty)
            self.assertEqual(str(clean.iloc[0]["operational_eligible"]).lower(), "false")
        finally:
            tmp.unlink(missing_ok=True)

    def test_operational_eligible_only_with_source_approved(self) -> None:
        ev = IntermediateEvent(
            title="non-farm employment change",
            country="United States",
            currency="USD",
            local_date_ny="2025-01-03",
            local_time_ny="08:30",
            source="bls_rule",
            source_type="official_rule",
            source_url="https://www.bls.gov",
            anchor_group="NFP",
        )
        clean0, _, _ = build_canonical_dataframe([ev], source_approved=False)
        clean1, _, _ = build_canonical_dataframe([ev], source_approved=True)
        self.assertEqual(str(clean0.iloc[0]["operational_eligible"]).lower(), "false")
        self.assertEqual(str(clean1.iloc[0]["operational_eligible"]).lower(), "true")


if __name__ == "__main__":
    unittest.main()
