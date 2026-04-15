from __future__ import annotations

import json
import shutil
import unittest
import uuid
import zoneinfo
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from research_lab.config import NY_TZ, NewsConfig
from research_lab.news_tradingeconomics import import_tradingeconomics_calendar


class NewsIngestaV2Tests(unittest.TestCase):
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

    def _write_json(self, path: Path, payload: list[dict[str, object]]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_v2_schema_fields_presence(self) -> None:
        """Verifica que el dataset resultante cumpla estrictamente con los campos del Schema V2."""
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_input.json"
            self._write_json(raw_path, [{
                "CalendarId": "V2_001",
                "Date": "2025-01-10T13:30:00",
                "Country": "United States",
                "Category": "Non Farm Payrolls",
                "Importance": 3,
            }])
            output_path = tmp / "news_v2.csv"
            settings = NewsConfig(source_approved=False)
            
            result = import_tradingeconomics_calendar(raw_path, clean_output_path=output_path, settings=settings)
            
            # Campos obligatorios solicitados
            expected_fields = {
                "event_id", "source", "title", "country", "currency",
                "importance", "scheduled_at_utc", "scheduled_at_ny",
                "timezone_source", "source_approved", "status", "operational_eligible",
            }
            actual_fields = set(result.clean_frame.columns)
            for field in expected_fields:
                with self.subTest(field=field):
                    self.assertIn(field, actual_fields)

    def test_utc_to_ny_conversion_winter_summer(self) -> None:
        """Verifica la conversión UTC -> NY en horarios de invierno y verano."""
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_input.json"
            self._write_json(raw_path, [
                {
                    "CalendarId": "WINTER_001",
                    "Date": "2025-01-10T13:30:00Z", # Invierno (EST: -5)
                    "Country": "United States",
                    "Category": "Non Farm Payrolls",
                    "Importance": 3,
                },
                {
                    "CalendarId": "SUMMER_001",
                    "Date": "2025-07-10T12:30:00Z", # Verano (EDT: -4)
                    "Country": "United States",
                    "Category": "Non Farm Payrolls",
                    "Importance": 3,
                }
            ])
            settings = NewsConfig()
            result = import_tradingeconomics_calendar(raw_path, clean_output_path=tmp/"v2.csv", settings=settings)
            
            # Ambas deben resultar en 08:30 NY si no hay error de DST
            winter = result.clean_frame[result.clean_frame["title"] == "non-farm employment change"].iloc[0]
            summer = result.clean_frame[result.clean_frame["title"] == "non-farm employment change"].iloc[1]
            
            self.assertTrue(winter["scheduled_at_ny"].endswith("-05:00"))
            self.assertTrue(summer["scheduled_at_ny"].endswith("-04:00"))
            self.assertIn("08:30:00", winter["scheduled_at_ny"])
            self.assertIn("08:30:00", summer["scheduled_at_ny"])

    def test_strict_anchor_time_rejection(self) -> None:
        """Verifica que el sistema RECHACE (no corrija) eventos que no coinciden con su hora ancla."""
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_input.json"
            self._write_json(raw_path, [{
                "CalendarId": "BAD_TIME_001",
                "Date": "2025-01-10T14:00:00Z", # 09:00 NY (Debería ser 08:30 para NFP)
                "Country": "United States",
                "Category": "Non Farm Payrolls",
                "Importance": 3,
            }])
            settings = NewsConfig()
            result = import_tradingeconomics_calendar(raw_path, clean_output_path=tmp/"v2.csv", settings=settings)
            
            # El dataset limpio debe estar vacío
            self.assertEqual(len(result.clean_frame), 0)
            
            # El audit debe mostrar 'rejected_time_mismatch'
            audit = pd.DataFrame(result.audit_frame)
            self.assertEqual(audit.iloc[0]["status"], "rejected_time_mismatch")
            self.assertIn("expected_08:30_got_09:00_ny", audit.iloc[0]["notes"])

    def test_dst_conflict_week_usa_vs_eu(self) -> None:
        """
        Prueba el comportamiento en la semana de conflicto de DST (Marzo).
        EEUU cambia el 9 de Marzo 2025, Europa el 30 de Marzo 2025.
        """
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_input.json"
            # Evento el 14 de Marzo 2025 (EEUU ya cambió, EU no)
            self._write_json(raw_path, [{
                "CalendarId": "CONFLICT_001",
                "Date": "2025-03-14T12:30:00Z", # 08:30 NY (Porque US es -4 ahora)
                "Country": "United States",
                "Category": "Non Farm Payrolls",
                "Importance": 3,
            }])
            settings = NewsConfig()
            result = import_tradingeconomics_calendar(raw_path, clean_output_path=tmp/"v2.csv", settings=settings)
            
            # Debe ser aprobado porque el parser derivó correctamente -04:00 basándose en IANA
            self.assertEqual(len(result.clean_frame), 1)
            row = result.clean_frame.iloc[0]
            self.assertTrue(row["scheduled_at_ny"].endswith("-04:00"))
            self.assertIn("08:30:00", row["scheduled_at_ny"])

    def test_fail_closed_source_approved_flag(self) -> None:
        """Verifica que el flag source_approved se propague correctamente desde la configuración."""
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_input.json"
            self._write_json(raw_path, [{"CalendarId": "F01", "Date": "2025-01-10T13:30:00Z", "Country": "US", "Category": "CPI", "Importance": 3}])
            
            settings_unapproved = NewsConfig(source_approved=False)
            result_un = import_tradingeconomics_calendar(raw_path, clean_output_path=tmp/"un.csv", settings=settings_unapproved)
            self.assertFalse(result_un.clean_frame.iloc[0]["source_approved"])
            self.assertEqual(str(result_un.clean_frame.iloc[0]["operational_eligible"]).lower(), "false")

            settings_approved = NewsConfig(source_approved=True)
            result_ap = import_tradingeconomics_calendar(raw_path, clean_output_path=tmp/"ap.csv", settings=settings_approved)
            self.assertTrue(result_ap.clean_frame.iloc[0]["source_approved"])
            self.assertEqual(str(result_ap.clean_frame.iloc[0]["operational_eligible"]).lower(), "true")

    def test_duplicate_rejection(self) -> None:
        """Verifica que se rechacen duplicados basados en CalendarId o (source+title+currency+utc)."""
        with self._workspace_tempdir() as tmp:
            raw_path = tmp / "te_input.json"
            item = {"CalendarId": "DUP_01", "Date": "2025-01-10T13:30:00Z", "Country": "United States", "Category": "Non Farm Payrolls", "Importance": 3}
            self._write_json(raw_path, [item, item])
            
            settings = NewsConfig()
            result = import_tradingeconomics_calendar(raw_path, clean_output_path=tmp/"v2.csv", settings=settings)
            
            self.assertEqual(len(result.clean_frame), 1)
            audit = pd.DataFrame(result.audit_frame)
            self.assertEqual(len(audit[audit["status"] == "rejected_duplicate"]), 1)

if __name__ == "__main__":
    unittest.main()
