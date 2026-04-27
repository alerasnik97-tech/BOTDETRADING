
import unittest
import pandas as pd

class TestNoLookahead(unittest.TestCase):
    def test_ema_no_lookahead(self):
        """Validar que el cálculo de EMA use solo datos pasados"""
        data = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        # Al calcular EMA para el índice 3 (valor 1.3), no debe conocer el 1.4
        ema = data.ewm(span=2, adjust=False).mean()
        
        # Simulamos un motor que itera
        for i in range(1, len(data)):
            curr_ema = ema.iloc[i-1] # El motor usa la EMA de la vela CERRADA anterior
            val_at_i = data.iloc[i]
            # Si curr_ema dependiera de val_at_i, habría lookahead
            # En pandas ewm, el valor en i incluye i. Por eso usamos i-1.
            self.assertNotEqual(curr_ema, ema.iloc[i], "El motor debe usar la vela cerrada para evitar lookahead")

    def test_fractal_delay(self):
        """Validar que un fractal N requiere N velas de confirmación posteriores"""
        # Fractal N=2 requiere vela en i+1 y i+2 para confirmarse
        # Si se detecta en i, es lookahead.
        def is_high_fractal(prices, i, n):
            if i < n or i >= len(prices) - n: return False
            center = prices[i]
            for j in range(i-n, i+n+1):
                if i == j: continue
                if prices[j] > center: return False
            return True

        prices = [1, 2, 5, 2, 1] # Fractal en index 2
        # A tiempo T=2 (valor 5), ¿podemos saber que es fractal? NO.
        # Solo en T=4 podemos confirmar.
        self.assertTrue(is_high_fractal(prices, 2, 2))
        
        for t in range(5):
            # Simulamos lo que ve el motor en cada paso de tiempo
            visible_prices = prices[:t+1]
            if t < 4: # Hasta que no llega el index 4, no se puede confirmar el fractal de index 2
                self.assertFalse(is_high_fractal(visible_prices, 2, 2), f"Fractal confirmado prematuramente en T={t}")

if __name__ == "__main__":
    unittest.main()
