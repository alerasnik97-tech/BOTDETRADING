
import unittest
import pandas as pd
import numpy as np
from phase18_h1_fractal_sweep import get_h1_fractals

class TestPhase18Lookahead(unittest.TestCase):
    def test_h1_fractal_no_lookahead(self):
        # Create dummy data: peak at index 5
        data = {
            'high_bid': [1.10, 1.11, 1.10, 1.11, 1.10, 1.20, 1.10, 1.11, 1.10, 1.11, 1.10],
            'low_bid':  [1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05]
        }
        df = pd.DataFrame(data)
        
        # N=2 fractal at index 5 needs confirmation from index 6 and 7
        # Confirmation bar should be index 7
        fh, fl = get_h1_fractals(df, n=2)
        
        # Check indices 0-6: should be NaN
        for i in range(7):
            self.assertTrue(np.isnan(fh[i]), f"Fractal visible too early at index {i}")
            
        # Check index 7: should be 1.20
        self.assertEqual(fh[7], 1.20, "Fractal not confirmed at expected index 7")
        
        # Check index 8+: should be NaN (unless ffill is applied)
        for i in range(8, len(df)):
            self.assertTrue(np.isnan(fh[i]), f"Fractal high present after confirmation index {i}")

    def test_h1_fractal_n3_lookahead(self):
        # N=3 needs 3 bars after. Peak at 5 -> Confirmation at 8
        data = {'high_bid': [1.1]*5 + [1.2] + [1.1]*10, 'low_bid': [1.0]*16}
        df = pd.DataFrame(data)
        fh, fl = get_h1_fractals(df, n=3)
        self.assertTrue(np.isnan(fh[7]))
        self.assertEqual(fh[8], 1.2)

if __name__ == '__main__':
    unittest.main()
