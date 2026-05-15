import os
import shutil
import sys
import tempfile
import unittest
import uuid
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import f06_rebuild_pipeline as pipeline


class TestPhase3RunnerAdversarial(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.created_output_dirs = []
        self.pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.allowed_report_root = os.path.abspath(os.path.join(
            self.pipeline_root, "..", "..", "reports", "f06_clean_train_only_rerun"))
        self.good_cfg_path = os.path.abspath(os.path.join(
            self.pipeline_root, "configs", "F06_PHASE3_CLEAN_TRAIN_ONLY.yaml"))
        with open(self.good_cfg_path, "r", encoding="utf-8") as f:
            self.good_cfg = f.read()

    def tearDown(self):
        for path in self.created_output_dirs:
            if os.path.exists(path):
                shutil.rmtree(path)
        shutil.rmtree(self.test_dir)

    def run_cmd(self, *args):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            exit_code = pipeline.main([str(a) for a in args])
            return exit_code, mock_stdout.getvalue()

    def allowed_output_path(self, label):
        path = os.path.join(self.allowed_report_root, f"unit_adv_{label}_{uuid.uuid4().hex}")
        self.created_output_dirs.append(path)
        return path

    def write_cfg(self, label, text):
        path = os.path.join(self.test_dir, f"{label}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def validate_bad_config(self, label, text, expected_error):
        cfg = self.write_cfg(label, text)
        code, out = self.run_cmd("validate_config", "--config", cfg)
        self.assertEqual(code, 2, out)
        self.assertIn("FAIL", out)
        self.assertIn(expected_error, out)

    def exact_months_block(self):
        return '''exact_months:
    - "2020-03"
    - "2021-08"
    - "2022-05"
    - "2023-01"
    - "2024-04"'''

    def test_validate_config_rejects_usdjpy(self):
        self.validate_bad_config(
            "usdjpy",
            self.good_cfg.replace("symbol: EURUSD", "symbol: USDJPY"),
            "symbol must be 'EURUSD'",
        )

    def test_validate_config_rejects_missing_session(self):
        text = self.good_cfg.replace(
            'session:\n  timezone: America/New_York\n  start: "07:00"\n  end: "17:00"\n',
            "",
        )
        self.validate_bad_config("missing_session", text, "session must be a mapping")

    def test_validate_config_rejects_missing_risk(self):
        text = self.good_cfg.replace("risk:\n  max_trades_per_day: 3\n", "")
        self.validate_bad_config("missing_risk", text, "risk must be a mapping")

    def test_validate_config_rejects_one_exact_month_only(self):
        text = self.good_cfg.replace(
            self.exact_months_block(),
            'exact_months:\n    - "2020-03"',
        )
        self.validate_bad_config("one_month", text, "data_scope.exact_months")

    def test_validate_config_rejects_wrong_month_order(self):
        text = self.good_cfg.replace(
            self.exact_months_block(),
            'exact_months:\n    - "2021-08"\n    - "2020-03"\n    - "2022-05"\n    - "2023-01"\n    - "2024-04"',
        )
        self.validate_bad_config("wrong_month_order", text, "data_scope.exact_months")

    def test_validate_config_rejects_output_dir_must_not_exist_false(self):
        self.validate_bad_config(
            "output_dir_false",
            self.good_cfg.replace("output_dir_must_not_exist: true", "output_dir_must_not_exist: false"),
            "output_rules.output_dir_must_not_exist",
        )

    def test_validate_config_rejects_max_trades_per_day_not_three(self):
        self.validate_bad_config(
            "max_trades",
            self.good_cfg.replace("max_trades_per_day: 3", "max_trades_per_day: 4"),
            "risk.max_trades_per_day",
        )

    def test_validate_config_rejects_session_start_not_0700(self):
        self.validate_bad_config(
            "bad_start",
            self.good_cfg.replace('start: "07:00"', 'start: "08:00"'),
            "session.start",
        )

    def test_validate_config_rejects_session_end_not_1700(self):
        self.validate_bad_config(
            "bad_end",
            self.good_cfg.replace('end: "17:00"', 'end: "18:00"'),
            "session.end",
        )

    def test_validate_config_rejects_timezone_not_new_york(self):
        self.validate_bad_config(
            "bad_timezone",
            self.good_cfg.replace("timezone: America/New_York", "timezone: UTC"),
            "session.timezone",
        )

    def test_prepare_phase3_run_rejects_existing_output_dir(self):
        out_dir = self.allowed_output_path("prepare_existing")
        os.mkdir(out_dir)
        code, out = self.run_cmd("prepare_phase3_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_PHASE3_PREP_FAILED", out)
        self.assertIn("already exists", out)

    def test_prepare_phase3_run_rejects_quarantined_path(self):
        out_dir = os.path.join(self.allowed_report_root, "QUARANTINED_DO_NOT_USE")
        code, out = self.run_cmd("prepare_phase3_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_PHASE3_PREP_FAILED", out)
        self.assertIn("quarantined token", out)

    def test_prepare_phase3_run_rejects_path_outside_allowed_reports(self):
        out_dir = os.path.join(self.test_dir, "outside_reports")
        code, out = self.run_cmd("prepare_phase3_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_PHASE3_PREP_FAILED", out)
        self.assertIn("outside allowed Phase 3 reports subtree", out)

    def test_dry_run_rejects_existing_output_dir(self):
        out_dir = self.allowed_output_path("dry_existing")
        os.mkdir(out_dir)
        code, out = self.run_cmd("dry_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_OUTPUT_DIR_EXISTS", out)

    def test_dry_run_rejects_quarantined_path(self):
        out_dir = os.path.join(self.allowed_report_root, "QUARANTINED_DO_NOT_USE")
        code, out = self.run_cmd("dry_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_FORBIDDEN_OUTPUT_PATH", out)

    def test_dry_run_rejects_emit_with_quarantined_token(self):
        out_dir = self.allowed_output_path("dry_emit_quarantine")
        forbidden = os.path.join(out_dir, "QUARANTINED_DO_NOT_USE.json")
        code, out = self.run_cmd(
            "dry_run",
            "--config",
            self.good_cfg_path,
            "--output-dir",
            out_dir,
            "--emit",
            "QUARANTINED_DO_NOT_USE.json",
        )
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_FORBIDDEN_OUTPUT_PATH", out)
        self.assertFalse(os.path.exists(forbidden), forbidden)
        self.assertFalse(os.path.exists(out_dir), out_dir)

    def test_dry_run_rejects_emit_legacy_v50b_name(self):
        out_dir = self.allowed_output_path("dry_emit_v50b")
        forbidden = os.path.join(out_dir, "V50B_RERUN_TRADES.csv")
        code, out = self.run_cmd(
            "dry_run",
            "--config",
            self.good_cfg_path,
            "--output-dir",
            out_dir,
            "--emit",
            "V50B_RERUN_TRADES.csv",
        )
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_FORBIDDEN_OUTPUT_PATH", out)
        self.assertFalse(os.path.exists(forbidden), forbidden)
        self.assertFalse(os.path.exists(out_dir), out_dir)

    def test_dry_run_rejects_emit_path_traversal(self):
        out_dir = self.allowed_output_path("dry_emit_traversal")
        escaped = os.path.abspath(os.path.join(out_dir, os.pardir, "MANIFEST.json"))
        code, out = self.run_cmd(
            "dry_run",
            "--config",
            self.good_cfg_path,
            "--output-dir",
            out_dir,
            "--emit",
            "../MANIFEST.json",
        )
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_FORBIDDEN_OUTPUT_PATH", out)
        self.assertIn("emit path escapes dry_run output_dir", out)
        self.assertFalse(os.path.exists(escaped), escaped)
        self.assertFalse(os.path.exists(out_dir), out_dir)

    def test_dry_run_does_not_create_nested_project_tree_under_pipeline(self):
        nested = os.path.join(self.pipeline_root, "03_RESEARCH_LAB")
        self.assertFalse(os.path.exists(nested), nested)
        out_dir = self.allowed_output_path("dry_no_nested")
        code, out = self.run_cmd("dry_run", "--config", self.good_cfg_path, "--output-dir", out_dir)
        self.assertEqual(code, 0, out)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "DRYRUN_MANIFEST.json")))
        self.assertFalse(os.path.exists(nested), nested)

    def test_run_phase3_still_blocks_without_explicit_confirmation(self):
        code, out = self.run_cmd("run_phase3", "--config", self.good_cfg_path, "--output-dir", self.allowed_output_path("run_no_confirm"))
        self.assertEqual(code, 2, out)
        self.assertIn("BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION", out)

    def test_run_phase3_with_confirmation_still_blocks_missing_adapter(self):
        code, out = self.run_cmd(
            "run_phase3",
            "--config",
            self.good_cfg_path,
            "--output-dir",
            self.allowed_output_path("run_with_confirm"),
            "--confirm-real-run",
            "PHASE3_F06_TRAIN_ONLY_APPROVED",
        )
        self.assertEqual(code, 2, out)
        self.assertIn("NOT_IMPLEMENTED_FAIL_CLOSED", out)


if __name__ == "__main__":
    unittest.main()
