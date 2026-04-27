
import unittest
from datetime import time

class TestTimeBoundaries(unittest.TestCase):
    def test_end_time_exclusive(self):
        """Validar que la hora de fin sea exclusiva"""
        end_time = time(11, 0)
        
        # 10:59 -> Permitido
        self.assertTrue(time(10, 59) < end_time)
        
        # 11:00 -> Bloqueado
        self.assertFalse(time(11, 0) < end_time)

    def test_rollover_block(self):
        """Validar bloqueo de rollover 17:00-19:00 NY"""
        def is_in_rollover(t):
            # 17:00 a 19:00
            start = time(17, 0)
            end = time(19, 0)
            return start <= t < end
        
        self.assertTrue(is_in_rollover(time(17, 30)))
        self.assertTrue(is_in_rollover(time(17, 0)))
        self.assertFalse(is_in_rollover(time(16, 59)))
        self.assertFalse(is_in_rollover(time(19, 0)))

if __name__ == "__main__":
    unittest.main()
