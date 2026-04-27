import csv
import sys
import tempfile
import unittest
from pathlib import Path

# Añadir directorio monitoring al path
sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring"))

from operational_analytics import analyze_log


CSV_HEADER = [
    "date",
    "feeder_status",
    "autopilot_status",
    "promotion_status",
    "chain_status",
    "classification",
    "coverage_ready",
    "daily_operable",
    "blockers",
    "bundle_updated",
]


class OperationalAnalyticsTests(unittest.TestCase):
    def test_analyze_log_builds_metrics_streaks_and_alerts(self):
        rows = [
            [
                "2026-04-20",
                "PRESENT",
                "OK",
                "M5:STAGING",
                "BLOCKED",
                "DATA_REFRESH_CANONICAL",
                "True",
                "True",
                "NONE",
                "True",
            ],
            [
                "2026-04-21",
                "PRESENT",
                "OK",
                "M5:BLOCK",
                "BLOCKED",
                "AUTOMATION_BLOCKED_BY_REAL_ERROR",
                "False",
                "False",
                "SCHEMA_INVALID",
                "False",
            ],
            [
                "2026-04-22",
                "PRESENT",
                "OK",
                "M5:BLOCK",
                "BLOCKED",
                "AUTOMATION_BLOCKED_BY_REAL_ERROR",
                "False",
                "False",
                "SCHEMA_INVALID",
                "False",
            ],
            [
                "2026-04-23",
                "PRESENT",
                "OK",
                "M5:BLOCK",
                "BLOCKED",
                "FAIL_CLOSED_CORRECT_BEHAVIOR",
                "False",
                "False",
                "NO_TARGET_ROWS",
                "False",
            ],
            [
                "2026-04-24",
                "PRESENT",
                "OK",
                "M5:PROMOTED",
                "SUCCESS",
                "DAILY_CHAIN_EXECUTED",
                "True",
                "True",
                "NONE",
                "True",
            ],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "daily_operational_log.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(CSV_HEADER)
                writer.writerows(rows)

            report = analyze_log(csv_path)

        self.assertEqual(report["metrics"]["total_runs"], 5)
        self.assertEqual(report["metrics"]["unique_dates"], 5)
        self.assertAlmostEqual(report["metrics"]["success_rate_operational"], 0.6)
        self.assertAlmostEqual(report["metrics"]["error_rate_real"], 0.4)
        self.assertAlmostEqual(report["metrics"]["fail_closed_rate"], 0.2)
        self.assertAlmostEqual(report["metrics"]["canonical_overlap_rate"], 0.2)
        self.assertAlmostEqual(report["metrics"]["coverage_ready_rate"], 0.4)
        self.assertAlmostEqual(report["metrics"]["daily_operable_rate"], 0.4)
        self.assertEqual(
            report["streaks"]["current_streak_without_real_errors"]["current_length"],
            2,
        )

        alert_codes = {alert["code"] for alert in report["alerts"]}
        self.assertIn("REAL_ERROR_STREAK", alert_codes)
        self.assertIn("COVERAGE_READY_DROP", alert_codes)
        self.assertIn("BUNDLE_NOT_UPDATED_STREAK", alert_codes)
        self.assertEqual(
            report["patterns"]["repetitive_blockers"][0]["blocker"],
            "SCHEMA_INVALID",
        )

    def test_missing_required_columns_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "broken.csv"
            csv_path.write_text("date,classification\n2026-04-20,OK\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                analyze_log(csv_path)


if __name__ == "__main__":
    unittest.main()
