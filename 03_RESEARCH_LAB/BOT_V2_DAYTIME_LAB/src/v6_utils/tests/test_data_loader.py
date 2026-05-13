
import unittest
import sys
import os
import pandas as pd
import numpy as np
import pytest

# Asegurar que el path incluya v6_utils
sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")

from v6_utils.data_loader import iter_months, load_month, iter_ticks_chunked, load_range_bulk
from v6_utils.memory import safe_collect

class TestDataLoader(unittest.TestCase):
    def test_iter_months_count(self):
        months = list(iter_months("2015-01", "2026-04"))
        # 11 años (2015-2025) * 12 + 4 meses (2026) = 132 + 4 = 136
        self.assertEqual(len(months), 136)

    @pytest.mark.xfail(reason="LEGACY_TEST_PATH_EXPECTATION: Ruta de parquets migrada a 05_MARKET_DATA_VAULT")
    def test_load_month_schema(self):
        # Usar marzo 2026 como muestra
        df = load_month(2026, 3, columns=["bid", "ask"])
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.index.tz, pd.Timestamp("2026-01-01").tz_localize("UTC").tzinfo)
        self.assertIn("bid", df.columns)
        self.assertIn("ask", df.columns)

    @pytest.mark.xfail(reason="LEGACY_TEST_PATH_EXPECTATION: Ruta de parquets migrada a 05_MARKET_DATA_VAULT")
    def test_load_month_selective_cols(self):
        df = load_month(2026, 3, columns=["bid"])
        self.assertEqual(list(df.columns), ["bid"])

    @pytest.mark.xfail(reason="LEGACY_TEST_PATH_EXPECTATION: Ruta de parquets migrada a 05_MARKET_DATA_VAULT")
    def test_load_month_downcast_safe(self):
        df_no = load_month(2026, 3, columns=["bid"], downcast_floats=False)
        df_yes = load_month(2026, 3, columns=["bid"], downcast_floats=True)
        
        self.assertEqual(df_no["bid"].dtype, "float64")
        self.assertEqual(df_yes["bid"].dtype, "float32")
        
        # Diferencia máxima (float32 tiene suficiente precisión para 5 decimales)
        diff = (df_no["bid"] - df_yes["bid"].astype("float64")).abs().max()
        self.assertLess(diff, 0.000005)

    @pytest.mark.xfail(reason="LEGACY_TEST_PATH_EXPECTATION: Ruta de parquets migrada a 05_MARKET_DATA_VAULT")
    def test_iter_ticks_chunked_releases_ram(self):
        from v6_utils.memory import get_process_rss_mb
        
        rss_start = get_process_rss_mb()
        # Iterar 3 meses (para no tardar mucho en el test)
        for df in iter_ticks_chunked("2026-01", "2026-03", columns=["bid"]):
            self.assertGreater(len(df), 0)
            del df
            safe_collect()
            
        rss_end = get_process_rss_mb()
        # El delta debería ser pequeño si se liberó correctamente
        self.assertLess(rss_end - rss_start, 200.0)

    def test_load_range_bulk_respects_budget(self):
        # Forzar error con budget minúsculo
        with self.assertRaises(MemoryError):
            load_range_bulk("2026-01", "2026-03", columns=["bid"], max_budget_mb=5)

if __name__ == '__main__':
    unittest.main()
