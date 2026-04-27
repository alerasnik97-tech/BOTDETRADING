
import unittest
from datetime import datetime, timedelta

class TestNewsGuard(unittest.TestCase):
    def is_blocked(self, trade_time, news_time, buffer_mins):
        diff = abs((trade_time - news_time).total_seconds() / 60)
        return diff < buffer_mins

    def test_news_boundary(self):
        """Validar el bloqueo de noticias en la frontera de ±30m"""
        news_time = datetime(2026, 4, 27, 8, 30) # 8:30 USD News
        buffer = 30
        
        # 8:01 -> Bloqueado (29m antes)
        self.assertTrue(self.is_blocked(datetime(2026, 4, 27, 8, 1), news_time, buffer))
        
        # 8:00 -> Permitido (30m antes, frontera inclusiva/exclusiva según política)
        # Usamos < 30, así que 30m exactos está permitido
        self.assertFalse(self.is_blocked(datetime(2026, 4, 27, 8, 0), news_time, buffer))
        
        # 8:59 -> Bloqueado (29m después)
        self.assertTrue(self.is_blocked(datetime(2026, 4, 27, 8, 59), news_time, buffer))
        
        # 9:01 -> Permitido
        self.assertFalse(self.is_blocked(datetime(2026, 4, 27, 9, 1), news_time, buffer))

if __name__ == "__main__":
    unittest.main()
