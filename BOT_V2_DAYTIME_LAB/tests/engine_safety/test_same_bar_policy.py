
import unittest

class TestSameBarPolicy(unittest.TestCase):
    def test_conservative_same_bar(self):
        """Si en la misma vela se tocan TP y SL, debe asumirse SL (política conservadora)"""
        entry = 1.0800
        sl = 1.0790
        tp = 1.0820
        
        # Vela de alta volatilidad: Low 1.0780, High 1.0830
        candle_low = 1.0780
        candle_high = 1.0830
        
        def resolve_trade(low, high, sl_p, tp_p):
            hit_sl = low <= sl_p
            hit_tp = high >= tp_p
            
            if hit_sl and hit_tp:
                return "SL" # Política conservadora obligatoria
            elif hit_sl:
                return "SL"
            elif hit_tp:
                return "TP"
            return "OPEN"

        result = resolve_trade(candle_low, candle_high, sl, tp)
        self.assertEqual(result, "SL", "En conflicto de same-bar, el motor debe penalizar (asumir SL)")

if __name__ == "__main__":
    unittest.main()
