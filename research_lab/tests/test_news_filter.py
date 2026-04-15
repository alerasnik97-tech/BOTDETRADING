from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.config import NewsConfig, NY_TZ
from research_lab.engine import entry_open_index
from research_lab.news_filter import build_entry_block, build_news_datasets, load_news_events


class NewsFilterTests(unittest.TestCase):
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

    def _write_news(self, path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def _settings(self, tmp: Path, raw_name: str = "raw_news.csv") -> NewsConfig:
        return NewsConfig(
            enabled=True,
            file_path=tmp / "validated_news.csv",
            raw_file_path=tmp / raw_name,
            source_approved=True,
            pre_minutes=15,
            post_minutes=15,
            currencies=("USD", "EUR"),
        )

    def test_supported_fixed_time_event_is_corrected_to_expected_ny_time(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T00:00:00+03:30", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = self._settings(tmp)
            result = load_news_events("EURUSD", settings)
            self.assertTrue(result.enabled)
            self.assertEqual(len(result.events), 1)
            self.assertEqual(result.events.iloc[0]["event_name_normalized"], "non-farm employment change")
            self.assertEqual(pd.Timestamp(result.events.iloc[0]["timestamp_ny"]).tz_convert(NY_TZ).strftime("%Y-%m-%d %H:%M"), "2025-01-10 08:30")
            self.assertGreaterEqual(result.suspicious_fixed_time_events, 1)

    def test_supported_raw_timestamp_preserves_expected_utc_and_ny_time(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T17:00:00+03:30", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = self._settings(tmp)
            result = load_news_events("EURUSD", settings)
            self.assertTrue(result.enabled)
            row = result.events.iloc[0]
            self.assertEqual(pd.Timestamp(row["timestamp_utc"]).strftime("%Y-%m-%d %H:%M"), "2025-01-10 13:30")
            self.assertEqual(pd.Timestamp(row["timestamp_ny"]).tz_convert(NY_TZ).strftime("%Y-%m-%d %H:%M"), "2025-01-10 08:30")

    def test_duplicate_rows_are_deduplicated_in_clean_dataset(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T17:00:00+03:30", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                    {"DateTime": "2025-01-10T17:00:00+03:30", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = self._settings(tmp)
            clean_frame, audit_frame, diagnostics = build_news_datasets("EURUSD", settings, start="2025-01-01", end="2025-01-31")
            self.assertEqual(len(clean_frame), 1)
            self.assertGreaterEqual(diagnostics["duplicate_rows_removed"], 1)
            self.assertEqual(int((audit_frame["validation_status"] == "rejected_duplicate").sum()), 1)

    def test_non_high_impact_rows_are_rejected(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T17:00:00+03:30", "Currency": "USD", "Impact": "Low Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = self._settings(tmp)
            clean_frame, audit_frame, _ = build_news_datasets("EURUSD", settings, start="2025-01-01", end="2025-01-31")
            self.assertEqual(len(clean_frame), 0)
            self.assertEqual(audit_frame.iloc[0]["validation_status"], "rejected_impact_level")

    def test_unsupported_event_disables_module_when_no_approved_rows(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T16:00:00+00:00", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Fed Chair Powell Speaks"},
                ],
            )
            settings = self._settings(tmp)
            result = load_news_events("EURUSD", settings)
            self.assertFalse(result.enabled)
            self.assertEqual(result.disabled_reason, "no_approved_events")

    def test_dst_correction_uses_expected_new_york_offset(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-07-03T00:00:00+04:30", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = self._settings(tmp)
            result = load_news_events("EURUSD", settings)
            row = result.events.iloc[0]
            self.assertEqual(pd.Timestamp(row["timestamp_ny"]).tz_convert(NY_TZ).strftime("%Y-%m-%d %H:%M %z"), "2025-07-03 08:30 -0400")

    def test_entry_block_uses_execution_timestamps(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T17:00:00+03:30", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = self._settings(tmp)
            result = load_news_events("EURUSD", settings)
            index = pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-01-10 08:15:00", tz=NY_TZ),
                    pd.Timestamp("2025-01-10 08:30:00", tz=NY_TZ),
                    pd.Timestamp("2025-01-10 08:45:00", tz=NY_TZ),
                ]
            )
            mask = build_entry_block(entry_open_index(index), result.events, settings)
            self.assertEqual(mask.tolist(), [False, True, True])

    def test_source_not_approved_disables_module_even_with_clean_rows(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T17:00:00+03:30", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = NewsConfig(
                enabled=True,
                file_path=tmp / "validated_news.csv",
                raw_file_path=raw_path,
                source_approved=False,
                pre_minutes=15,
                post_minutes=15,
                currencies=("USD", "EUR"),
            )
            result = load_news_events("EURUSD", settings)
            self.assertFalse(result.enabled)
            self.assertEqual(result.disabled_reason, "source_not_approved")
            self.assertGreater(result.approved_rows, 0)
            self.assertEqual(len(result.events), 0)

    def test_force_enabled_with_unapproved_source_fails_closed_and_blocks_leaks(self) -> None:
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2025-01-10T08:30:00+00:00", "Currency": "USD", "Impact": "High Impact Expected", "Event": "Non-Farm Employment Change"},
                ],
            )
            settings = NewsConfig(
                enabled=True,  # Usuario intenta forzar prendido
                file_path=tmp / "validated_news.csv",
                raw_file_path=raw_path,
                source_approved=False,  # Pero la fuente (Forex Factory) esta vetada
                pre_minutes=15,
                post_minutes=15,
                currencies=("USD", "EUR"),
            )
            result = load_news_events("EURUSD", settings)
            # CONFIRMACION FAIL-CLOSED REAL
            self.assertFalse(result.enabled)
            self.assertEqual(len(result.events), 0)
            self.assertEqual(result.disabled_reason, "source_not_approved")

    def test_dst_gap_between_europe_and_us_would_misalign_hardcoded_ecb_schedule(self) -> None:
        """
        Este test demuestra la debilidad matematica que invalida el modulo.
        Europa y USA cambian de DST en domingos diferentes de marzo y octubre.
        ECB Press Conference es a las 14:45 Frankfurt time.
        Eso es 08:45 NY en invierno/verano emparejado, PERO es 09:45 NY durante el GAP de DST.
        El modulo asume erroneamente 08:30 FIJO.
        """
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "raw_news.csv"
            self._write_news(
                raw_path,
                [
                    {"DateTime": "2024-03-14T00:00:00", "Currency": "EUR", "Impact": "High Impact Expected", "Event": "ECB Press Conference"},
                ],
            )
            settings = self._settings(tmp)
            result = load_news_events("EURUSD", settings)
            if len(result.events) > 0:
                row = result.events.iloc[0]
                ny_time = pd.Timestamp(row["timestamp_ny"]).tz_convert(NY_TZ).strftime("%H:%M")
                # El modulo lo clava a las 08:30 por hardcode (aprobado_fixed_schedule)
                self.assertEqual(ny_time, "08:30")
                # PERO EN LA REALIDAD ES 09:45 porque US cambio a DST el 10/Mar y EU no cambia hasta el 31/Mar!
                # Este false-positive es inaceptable operativamente.
