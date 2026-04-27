
import unittest

class TestCostSensitivity(unittest.TestCase):
    def simulate_pnl(self, trades, spread, slippage):
        # trades: list of (r_expected)
        # simplistic model: costs reduce R
        cost_r = (spread + slippage) * 10000 / 10 # asumimos risk unitario de 10 pips
        return sum([t - cost_r if t > 0 else t - cost_r for t in trades])

    def test_costs_degrade_performance(self):
        """Validar que el aumento de costos nunca mejore artificialmente el resultado"""
        trades = [2.0, -1.0, 2.0, 2.0, -1.0] # Historial sintético
        
        base_pnl = self.simulate_pnl(trades, 0.00007, 0.0)
        high_cost_pnl = self.simulate_pnl(trades, 0.00015, 0.00005)
        
        self.assertLess(high_cost_pnl, base_pnl, "El aumento de costos debe degradar el PnL")

if __name__ == "__main__":
    unittest.main()
