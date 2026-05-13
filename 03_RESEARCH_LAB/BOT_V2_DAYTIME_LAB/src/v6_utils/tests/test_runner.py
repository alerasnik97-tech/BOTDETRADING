
import unittest
import pandas as pd
import sys
import os
from zoneinfo import ZoneInfo

# Asegurar que el path incluya v6_utils
sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")

from v6_utils.temporal import sanitize_utc_index, to_ny, session_anchor, is_market_open
from v6_utils.numeric import snap_to_tick, safe_add, pip_to_price

class TestTemporal(unittest.TestCase):
    def test_spring_forward_2024(self):
        ts_utc = pd.to_datetime(["2024-03-10 07:30:00"], utc=True)
        df = pd.DataFrame(index=ts_utc)
        df_ny = to_ny(df)
        self.assertEqual(df_ny.index[0].hour, 3)
        self.assertEqual(df_ny.index[0].minute, 30)

    def test_fall_back_2024(self):
        ts_utc = pd.to_datetime(["2024-11-03 05:30:00"], utc=True)
        df = pd.DataFrame(index=ts_utc)
        df_ny = to_ny(df)
        self.assertEqual(df_ny.index[0].hour, 1)
        self.assertEqual(df_ny.index[0].minute, 30)

    def test_session_anchor_evening(self):
        ny_tz = ZoneInfo("America/New_York")
        ts1 = pd.Timestamp("2024-05-14 22:00:00", tz=ny_tz)
        anchor1 = session_anchor(ts1)
        self.assertEqual(anchor1.hour, 17)
        self.assertEqual(anchor1.day, 14)

    def test_holidays_2024(self):
        ny_tz = ZoneInfo("America/New_York")
        ts = pd.Timestamp("2024-11-28 10:00:00", tz=ny_tz)
        self.assertFalse(is_market_open(ts))

    def test_no_dst_holes_in_data(self):
        # Usamos una muestra pequeña manual que simule el salto DST
        # March 10, 2024. 01:59 AM -> 03:00 AM
        # 06:59 UTC -> 01:59 EST
        # 07:00 UTC -> 03:00 EDT
        idx = pd.to_datetime(["2024-03-10 06:59:00", "2024-03-10 07:00:00"], utc=True)
        df = pd.DataFrame(index=idx)
        df_ny = to_ny(df)
        # No debería haber gap mayor a 1h en NY (el salto de 1h es lo esperado en reloj de pared)
        # Pero en timestamps UTC continuos, el salto es transparente.
        from v6_utils.temporal import assert_no_dst_holes
        assert_no_dst_holes(df_ny)

    def test_market_open_weekend(self):
        ny_tz = ZoneInfo("America/New_York")
        ts_sat = pd.Timestamp("2024-05-18 10:00:00", tz=ny_tz)
        self.assertFalse(is_market_open(ts_sat))
        ts_sun_open = pd.Timestamp("2024-05-19 18:00:00", tz=ny_tz)
        self.assertTrue(is_market_open(ts_sun_open))

class TestNumeric(unittest.TestCase):
    def test_snap_basic(self):
        self.assertEqual(snap_to_tick(1.0245678), 1.02457)

    def test_snap_no_drift(self):
        price = 1.10000
        for _ in range(1000):
            price = safe_add(price, 0.00001)
        self.assertAlmostEqual(price, 1.11, places=5)

    def test_banker_rounding_avoided(self):
        self.assertEqual(snap_to_tick(1.024565), 1.02457)

    def test_safe_add_commutative(self):
        self.assertEqual(safe_add(1.123456, 0.000014), safe_add(0.000014, 1.123456))

    def test_pip_to_price(self):
        self.assertEqual(pip_to_price(10), 0.001)

    def test_r_to_pips(self):
        from v6_utils.numeric import r_to_pips
        self.assertEqual(r_to_pips(2.0, 10.0), 20.0)

if __name__ == '__main__':
    unittest.main()
