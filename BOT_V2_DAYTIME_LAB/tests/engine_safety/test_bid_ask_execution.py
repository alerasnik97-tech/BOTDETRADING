
import unittest
import pandas as pd

class TestBidAskExecution(unittest.TestCase):
    def setUp(self):
        self.spread = 0.00010 # 1 pip

    def test_long_entry_ask(self):
        """LONG entra a ASK (Bid + Spread)"""
        bid_price = 1.0800
        ask_price = bid_price + self.spread
        entry_price = bid_price + self.spread # Simulamos lógica de entrada
        self.assertEqual(entry_price, 1.0801, "Long entry debe pagar el spread")

    def test_short_exit_ask(self):
        """SHORT sale a ASK (Bid + Spread)"""
        # Entrada Short a Bid
        entry_p = 1.0800
        sl_bid = 1.0810
        
        # Simulación de vela que toca SL
        curr_bid_high = 1.0809
        curr_ask_high = curr_bid_high + self.spread # 1.0810
        
        # El SL de Short se evalúa contra el ASK
        is_hit = curr_ask_high >= sl_bid
        self.assertTrue(is_hit, "SL de Short debe ser tocado por el ASK")
        
        # El SL de Short NO debe ser tocado si solo el BID llega
        sl_bid_higher = 1.0811
        is_hit_low = curr_ask_high >= sl_bid_higher
        self.assertFalse(is_hit_low, "SL de Short no debe tocarse si el ASK no llega")

    def test_long_exit_bid(self):
        """LONG sale a BID (High/Low Bid)"""
        entry_p = 1.0801 # (1.0800 + spread)
        tp_bid = 1.0821 # +20 pips
        
        # Vela llega a 1.0820 Bid
        curr_bid_high = 1.0820
        is_hit = curr_bid_high >= tp_bid
        self.assertFalse(is_hit, "TP de Long no debe tocarse si el BID no llega")
        
        # Vela llega a 1.0821 Bid
        curr_bid_high_hit = 1.0821
        is_hit_ok = curr_bid_high_hit >= tp_bid
        self.assertTrue(is_hit_ok, "TP de Long debe tocarse por el BID")

if __name__ == "__main__":
    unittest.main()
