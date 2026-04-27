from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "h6_paper_shadow_runner.py"
SPEC = importlib.util.spec_from_file_location("h6_paper_shadow_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H6PaperShadowRunnerTests(unittest.TestCase):
    def test_within_paper_window_is_strict_10_to_11(self) -> None:
        self.assertFalse(MODULE.within_paper_window(pd.Timestamp("2025-10-01 09:57:00", tz=MODULE.NY_TZ)))
        self.assertTrue(MODULE.within_paper_window(pd.Timestamp("2025-10-01 10:00:00", tz=MODULE.NY_TZ)))
        self.assertTrue(MODULE.within_paper_window(pd.Timestamp("2025-10-01 10:57:00", tz=MODULE.NY_TZ)))
        self.assertFalse(MODULE.within_paper_window(pd.Timestamp("2025-10-01 11:00:00", tz=MODULE.NY_TZ)))

    def test_filter_paper_signals_keeps_only_10_to_11(self) -> None:
        signals = pd.DataFrame(
            {
                "signal_time": [
                    pd.Timestamp("2025-07-28 08:27:00", tz=MODULE.NY_TZ),
                    pd.Timestamp("2025-10-01 10:12:00", tz=MODULE.NY_TZ),
                    pd.Timestamp("2025-11-14 08:48:00", tz=MODULE.NY_TZ),
                ],
                "direction": ["long", "short", "short"],
            }
        )
        filtered = MODULE.filter_paper_signals(signals)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["signal_time"].strftime("%Y-%m-%d %H:%M"), "2025-10-01 10:12")

    def test_cooldown_block_activates_after_three_losses_within_48h(self) -> None:
        ledger = pd.DataFrame(
            [
                {
                    "session_date": "2025-10-01",
                    "event_timestamp": "2025-10-01 11:18:00-0400",
                    "event_type": "PAPER_EXIT",
                    "result": "loss",
                },
                {
                    "session_date": "2025-10-02",
                    "event_timestamp": "2025-10-02 11:12:00-0400",
                    "event_type": "PAPER_EXIT",
                    "result": "loss",
                },
                {
                    "session_date": "2025-10-03",
                    "event_timestamp": "2025-10-03 10:54:00-0400",
                    "event_type": "PAPER_EXIT",
                    "result": "loss",
                },
            ]
        )
        blocked, reason, details = MODULE.cooldown_block("2025-10-04", ledger)
        self.assertTrue(blocked)
        self.assertEqual(reason, "cooldown_3_losses_48h")
        self.assertIn("cooldown_until=", details)

    def test_cooldown_block_resets_after_non_loss(self) -> None:
        ledger = pd.DataFrame(
            [
                {
                    "session_date": "2025-10-01",
                    "event_timestamp": "2025-10-01 11:18:00-0400",
                    "event_type": "PAPER_EXIT",
                    "result": "loss",
                },
                {
                    "session_date": "2025-10-02",
                    "event_timestamp": "2025-10-02 11:12:00-0400",
                    "event_type": "PAPER_EXIT",
                    "result": "win",
                },
                {
                    "session_date": "2025-10-03",
                    "event_timestamp": "2025-10-03 10:54:00-0400",
                    "event_type": "PAPER_EXIT",
                    "result": "loss",
                },
            ]
        )
        blocked, reason, details = MODULE.cooldown_block("2025-10-04", ledger)
        self.assertFalse(blocked)
        self.assertEqual(reason, "")
        self.assertEqual(details, "")


if __name__ == "__main__":
    unittest.main()
