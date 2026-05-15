import unittest
import os
import shutil
import tempfile
import sys
import uuid
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import f06_rebuild_pipeline as pipeline

class TestPhase3Runner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.created_output_dirs = []
        self.pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.allowed_report_root = os.path.abspath(os.path.join(
            self.pipeline_root, "..", "..", "reports", "f06_clean_train_only_rerun"))
        self.good_cfg_path = os.path.join(self.test_dir, "good.yaml")
        with open(self.good_cfg_path, "w", encoding="utf-8") as f:
            f.write("""
project: Trading BOT
phase: F06_PHASE3_CLEAN_TRAIN_ONLY_RERUN
mode: TRAIN_ONLY
symbol: EURUSD
families: [F06]
session:
  timezone: America/New_York
  start: "07:00"
  end: "17:00"
risk:
  max_trades_per_day: 3
data_scope:
  allow_2025: false
  allow_2026: false
  validation_enabled: false
  holdout_enabled: false
  exact_months:
    - "2020-03"
    - "2021-08"
    - "2022-05"
    - "2023-01"
    - "2024-04"
input_rules:
  forbid_quarantined_paths: true
  forbid_legacy_v50b_outputs: true
  forbid_old_master_ranking: true
  forbid_old_trades_csv: true
output_rules:
  output_dir_must_not_exist: true
  single_run_id_only: true
  no_validation_columns_in_train_only: true
  manifest_required: true
  hashes_required: true
cost_model:
  require_real_spread_component: true
  require_slippage_component: true
  require_round_turn_commission: true
sample_size:
  min_trades_per_family: 100
  min_trades_per_month_for_reporting: 10
            """)

    def tearDown(self):
        for path in self.created_output_dirs:
            if os.path.exists(path):
                shutil.rmtree(path)
        shutil.rmtree(self.test_dir)

    def run_cmd(self, *args):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            exit_code = pipeline.main([str(a) for a in args])
            return exit_code, mock_stdout.getvalue()

    def allowed_output_path(self, label):
        path = os.path.join(self.allowed_report_root, f"unit_{label}_{uuid.uuid4().hex}")
        self.created_output_dirs.append(path)
        return path

    def cfg_text(self):
        with open(self.good_cfg_path, "r", encoding="utf-8") as f:
            return f.read()

    def write_bad_cfg(self, label, text):
        cfg = os.path.join(self.test_dir, f"{label}.yaml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(text)
        return cfg

    def test_preflight_phase3_passes_clean_config(self):
        code, out = self.run_cmd("preflight_phase3", "--config", self.good_cfg_path, "--output-dir", self.allowed_output_path("preflight"))
        self.assertEqual(code, 0)
        self.assertIn("PREFLIGHT_PHASE3_PASS", out)

    def test_preflight_phase3_blocks_validation_enabled(self):
        cfg = self.write_bad_cfg("validation_enabled", self.cfg_text().replace("validation_enabled: false", "validation_enabled: true"))
        code, out = self.run_cmd("preflight_phase3", "--config", cfg)
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED_PHASE3_PREFLIGHT_FAILED", out)

    def test_preflight_phase3_blocks_holdout_enabled(self):
        cfg = self.write_bad_cfg("holdout_enabled", self.cfg_text().replace("holdout_enabled: false", "holdout_enabled: true"))
        code, out = self.run_cmd("preflight_phase3", "--config", cfg)
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED_PHASE3_PREFLIGHT_FAILED", out)

    def test_preflight_phase3_blocks_2025_month(self):
        cfg = self.write_bad_cfg("month_2025", self.cfg_text().replace('    - "2024-04"', '    - "2024-04"\n    - "2025-01"'))
        code, out = self.run_cmd("preflight_phase3", "--config", cfg)
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED_PHASE3_PREFLIGHT_FAILED", out)

    def test_preflight_phase3_blocks_family_not_f06(self):
        cfg = self.write_bad_cfg("family_f08", self.cfg_text().replace("[F06]", "[F06, F08]"))
        code, out = self.run_cmd("preflight_phase3", "--config", cfg)
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED_PHASE3_PREFLIGHT_FAILED", out)

    def test_preflight_phase3_blocks_old_v50b_path(self):
        code, out = self.run_cmd("preflight_phase3", "--config", self.good_cfg_path, "--output-dir", "v50b_limited_real_gauntlet_rerun_sw")
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED_PHASE3_PREFLIGHT_FAILED", out)

    def test_preflight_phase3_blocks_quarantined_path(self):
        code, out = self.run_cmd("preflight_phase3", "--config", self.good_cfg_path, "--output-dir", "QUARANTINED_DO_NOT_USE")
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED_PHASE3_PREFLIGHT_FAILED", out)

    def test_prepare_phase3_run_creates_manifest_draft_only(self):
        out_dir = self.allowed_output_path("prepared_run")
        code, out = self.run_cmd("prepare_phase3_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertEqual(code, 0)
        self.assertIn("PHASE3_RUN_PREPARED", out)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "PRE_RUN_MANIFEST_DRAFT.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "COMMANDS_PLANNED.md")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "SAFETY_PRECHECK.md")))

    def test_prepare_phase3_run_does_not_create_trades_or_ranking(self):
        out_dir = self.allowed_output_path("prepared_run_2")
        self.run_cmd("prepare_phase3_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "TRADES.csv")))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "RANKING.csv")))

    def test_run_phase3_requires_explicit_confirmation(self):
        code, out = self.run_cmd("run_phase3", "--config", self.good_cfg_path, "--output-dir", "test")
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION", out)

    def test_run_phase3_blocks_without_safe_engine_adapter(self):
        code, out = self.run_cmd("run_phase3", "--config", self.good_cfg_path, "--output-dir", self.allowed_output_path("run_missing_adapter"), "--confirm-real-run", "PHASE3_F06_TRAIN_ONLY_APPROVED")
        self.assertEqual(code, 2)
        self.assertIn("NOT_IMPLEMENTED_FAIL_CLOSED", out)

    def test_phase3_output_contract_required_files(self):
        # We ensure output contract documentation exists. Tests against the schema already exist in test_validate.py
        contract_path = os.path.join(os.path.dirname(__file__), "..", "contracts", "PHASE3_OUTPUT_CONTRACT.md")
        self.assertTrue(os.path.exists(contract_path), "Contract must exist")
