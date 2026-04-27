
import unittest
from datetime import datetime

class TestSundayGapLoader(unittest.TestCase):
    def is_market_open(self, dt):
        # 0: Monday, 4: Friday, 5: Saturday, 6: Sunday
        weekday = dt.weekday()
        hour = dt.hour
        
        if weekday == 5: return False # Saturday closed
        if weekday == 4 and hour >= 17: return False # Friday 17:00 NY closed
        if weekday == 6 and hour < 17: return False # Sunday before 17:00 NY closed
        return True

    def test_market_hours(self):
        """Validar apertura y cierre del mercado FX (NY Time)"""
        # Viernes 16:59 -> Abierto
        self.assertTrue(self.is_market_open(datetime(2026, 4, 24, 16, 59)))
        # Viernes 17:01 -> Cerrado
        self.assertFalse(self.is_market_open(datetime(2026, 4, 24, 17, 1)))
        
        # Sábado -> Cerrado
        self.assertFalse(self.is_market_open(datetime(2026, 4, 25, 12, 0)))
        
        # Domingo 16:59 -> Cerrado
        self.assertFalse(self.is_market_open(datetime(2026, 4, 26, 16, 59)))
        # Domingo 17:01 -> Abierto
        self.assertTrue(self.is_market_open(datetime(2026, 4, 26, 17, 1)))

if __name__ == "__main__":
    unittest.main()
