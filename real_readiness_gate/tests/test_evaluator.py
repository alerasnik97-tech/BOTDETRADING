import unittest
import os
import json
import sys

# Agregar el directorio padre al path para importar el evaluator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluator

class TestReadinessEvaluator(unittest.TestCase):
    def test_policy_loading(self):
        self.assertIn("min_sample_size", evaluator.POLICY)
        self.assertEqual(evaluator.POLICY["min_pf"], 2.0)

    def test_evaluate_existing_variant(self):
        variant_id = "tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m"
        res = evaluator.evaluate_line(variant_id)
        self.assertEqual(res["variant_id"], variant_id)
        self.assertIn("gates", res)
        # Debe tener al menos los gates de research
        self.assertTrue(res["gates"]["sample_size"]["pass"])

    def test_scorecard_generation(self):
        variant_id = "tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m"
        res = evaluator.evaluate_line(variant_id)
        evaluator.generate_reports(res)
        
        self.assertTrue(os.path.exists(os.path.join(evaluator.GATE_DIR, "real_readiness_scorecard.json")))
        self.assertTrue(os.path.exists(os.path.join(evaluator.GATE_DIR, "real_readiness_scorecard.md")))
        self.assertTrue(os.path.exists(os.path.join(evaluator.GATE_DIR, "real_readiness_summary.txt")))

    def test_invalid_variant(self):
        res = evaluator.evaluate_line("non_existent_variant")
        self.assertEqual(res["verdict"], "NOT_READY")
        self.assertIn("VARIANT_NOT_FOUND_IN_RESEARCH", res["blockers"])

if __name__ == "__main__":
    unittest.main()
