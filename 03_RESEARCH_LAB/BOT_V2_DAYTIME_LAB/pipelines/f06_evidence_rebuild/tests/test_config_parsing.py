import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_pipeline, ROOT

P = load_pipeline()
TEMPLATE = os.path.join(ROOT, "configs", "F06_REBUILD_TRAIN_ONLY_TEMPLATE.yaml")
PHASE3_CONFIG = os.path.join(ROOT, "configs", "F06_PHASE3_CLEAN_TRAIN_ONLY.yaml")


class TestConfigParsing(unittest.TestCase):
    """Hardens the no-PyYAML fallback. The old _mini_yaml_load mis-nested
    'key:' + indented '- item' into {key:{key:[...]}}, which turned the
    2025/2026 guard into a fail-OPEN path. These tests exercise the
    fallback DIRECTLY regardless of whether PyYAML is installed."""

    def setUp(self):
        with open(TEMPLATE, "r", encoding="utf-8") as fh:
            self.text = fh.read()
        with open(PHASE3_CONFIG, "r", encoding="utf-8") as fh:
            self.phase3_text = fh.read()

    def test_mini_yaml_lists_are_real_lists(self):
        cfg = P._mini_yaml_load(self.text)
        self.assertEqual(cfg.get("families"), ["F06"],
                         f"families mis-parsed: {cfg.get('families')!r}")
        months = cfg.get("data_scope", {}).get("exact_months")
        self.assertIsInstance(months, list)
        self.assertEqual(len(months), 5)
        comps = cfg.get("cost_model", {}).get("components")
        self.assertIsInstance(comps, list)
        self.assertIn("spread_component", comps)

    def test_mini_yaml_safety_constants(self):
        cfg = P._mini_yaml_load(self.text)
        self.assertEqual(str(cfg.get("mode")), "TRAIN_ONLY")
        ds = cfg.get("data_scope", {})
        for k in ("allow_2025", "allow_2026", "validation_enabled",
                  "holdout_enabled"):
            self.assertEqual(ds.get(k), False, f"{k} must parse to False")

    def test_mini_yaml_no_2025_2026_in_months(self):
        cfg = P._mini_yaml_load(self.text)
        months = cfg.get("data_scope", {}).get("exact_months")
        ok, errs = P.check_no_2025_2026(months)
        self.assertTrue(ok, errs)

    def test_load_config_invariants_pass(self):
        cfg = P.load_config(PHASE3_CONFIG)
        self.assertEqual(P._config_invariants(cfg), [])

    def test_mini_yaml_invariants_pass_even_without_pyyaml(self):
        # _config_invariants must hold on the fallback-parsed config too
        cfg = P._mini_yaml_load(self.phase3_text)
        self.assertEqual(P._config_invariants(cfg), [])


if __name__ == "__main__":
    unittest.main()
