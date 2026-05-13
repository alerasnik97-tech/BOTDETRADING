
import unittest
import pandas as pd
import sys
import os

sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")
from v6_utils.bars import build_bars, get_bar_at, assert_no_partial_bars

class TestBars(unittest.TestCase):
    def setUp(self):
        # Crear ticks sintéticos: 100 ticks cada 2 seg -> 200 seg total
        # Start at 10:00:00
        idx = pd.date_range("2024-03-10 10:00:00", periods=100, freq="2s", tz="UTC")
        self.ticks = pd.DataFrame({
            "bid": [1.1000 + i*0.00001 for i in range(100)],
            "ask": [1.1002 + i*0.00001 for i in range(100)]
        }, index=idx)
        self.ticks["mid"] = (self.ticks["bid"] + self.ticks["ask"]) / 2

    def test_no_partial_bars(self):
        # Ticks terminan a las 10:03:18
        # Barras M1 cierran a las :01, :02, :03
        bars = build_bars(self.ticks, "M1", anchor="midnight_utc")
        # La última barra cerrada debe ser 10:03:00 (que cubre 10:02:00 a 10:03:00)
        # La barra que cierra a las 10:04:00 es parcial (solo tiene ticks hasta 10:03:18)
        self.assertEqual(bars.index.max(), pd.Timestamp("2024-03-10 10:03:00", tz="UTC"))
        self.assertEqual(len(bars), 3)

    def test_get_bar_at_last_closed(self):
        bars = build_bars(self.ticks, "M1", anchor="midnight_utc")
        # Consulta a las 10:02:30 -> debe dar la barra que cerró a las 10:02:00
        bar = get_bar_at(bars, pd.Timestamp("2024-03-10 10:02:30", tz="UTC"))
        self.assertEqual(bar.name, pd.Timestamp("2024-03-10 10:02:00", tz="UTC"))
        
    def test_fx_day_anchor(self):
        # D1 bars should anchor to 17:00 NY
        # 17:00 NY is 21:00 UTC (during EST) or 22:00 UTC (during EDT)
        # March 10 is tricky due to DST. Use May for stable anchor test.
        idx = pd.date_range("2024-05-01 00:00:00", periods=5000, freq="1min", tz="UTC")
        ticks = pd.DataFrame({"bid": 1.1000, "ask": 1.1002}, index=idx)
        bars = build_bars(ticks, "D1", anchor="fx_day")
        
        # Cada close_time debe ser las 17:00 NY
        from v6_utils.temporal import to_ny
        ny_index = to_ny(pd.DataFrame(index=bars.index)).index
        for ts in ny_index:
            self.assertEqual(ts.hour, 17)
            self.assertEqual(ts.minute, 0)

    def test_midnight_utc_anchor(self):
        bars = build_bars(self.ticks, "H1", anchor="midnight_utc")
        for ts in bars.index:
            self.assertEqual(ts.minute, 0)
            self.assertEqual(ts.second, 0)

    def test_m5_buckets(self):
        # Ticks a las 10:04:30 y 10:05:00 (para cerrar la barra)
        ticks = pd.DataFrame({"bid": 1.1, "ask": 1.1002}, 
                            index=pd.to_datetime(["2024-03-10 10:04:30", "2024-03-10 10:05:00"], utc=True))
        bars = build_bars(ticks, "M5", anchor="midnight_utc")
        # El close_time debe ser 10:05:00
        self.assertEqual(bars.index[0], pd.Timestamp("2024-03-10 10:05:00", tz="UTC"))

    def test_m3_buckets(self):
        # 361 ticks cada 10 seg -> llega justo a las 11:00:00 para cerrar la barra 20
        idx = pd.date_range("2024-01-01 10:00:00", periods=361, freq="10s", tz="UTC")
        ticks = pd.DataFrame({"bid": 1.1, "ask": 1.1002}, index=idx)
        bars = build_bars(ticks, "M3", anchor="midnight_utc")
        # 60 min / 3 min = 20 barras
        self.assertEqual(len(bars), 20)
        # La primera barra M3 cierra a las 10:03:00
        self.assertEqual(bars.index[0], pd.Timestamp("2024-01-01 10:03:00", tz="UTC"))

if __name__ == '__main__':
    unittest.main()
