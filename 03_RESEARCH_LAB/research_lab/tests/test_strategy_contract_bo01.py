from __future__ import annotations

import builtins
import importlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_PATH = "research_lab.strategies.BO01Strategy"


def _module():
    return importlib.import_module(MODULE_PATH)


def _target_index(frame: pd.DataFrame, date: str, hhmm: str) -> int:
    target = pd.Timestamp(f"{date} {hhmm}:00", tz="UTC").tz_convert(frame.index.tz)
    return int(frame.index.get_loc(target))


def _bo01_frame(date: str = "2024-03-12", *, hhmm: str = "07:00", tz: str = "UTC",
                direction: str = "long") -> tuple[pd.DataFrame, int]:
    index_utc = pd.date_range(pd.Timestamp(f"{date} 00:00:00", tz="UTC"), periods=145, freq="5min")
    index = index_utc.tz_convert(tz)
    frame = pd.DataFrame(index=index)
    frame["open"] = 1.1000
    frame["high"] = 1.1004
    frame["low"] = 1.0995
    frame["close"] = 1.1000
    frame["volume"] = 1000.0
    frame["atr14"] = 0.0004
    frame["ema_m15_200"] = 1.1000
    i = _target_index(frame, date, hhmm)
    if direction == "long":
        frame.iloc[i, frame.columns.get_loc("open")] = 1.1001
        frame.iloc[i, frame.columns.get_loc("high")] = 1.1009
        frame.iloc[i, frame.columns.get_loc("low")] = 1.1000
        frame.iloc[i, frame.columns.get_loc("close")] = 1.1008
    else:
        frame.iloc[i, frame.columns.get_loc("open")] = 1.0998
        frame.iloc[i, frame.columns.get_loc("high")] = 1.0999
        frame.iloc[i, frame.columns.get_loc("low")] = 1.0990
        frame.iloc[i, frame.columns.get_loc("close")] = 1.0991
    return frame, i


def _assert_signal_contract(testcase: unittest.TestCase, result: dict | None) -> None:
    if result is None:
        return
    testcase.assertIn(result["signal"], {-1, 1})
    testcase.assertIn(result["direction"], {"long", "short"})
    testcase.assertEqual(result["stop_mode"], "price")
    testcase.assertIn("stop_price", result)
    testcase.assertEqual(result["target_mode"], "rr")
    testcase.assertEqual(result["target_rr"], 2.0)
    testcase.assertIn("session_name", result)


class BO01ContractTests(unittest.TestCase):
    def test_import_and_module_contract(self) -> None:
        module = _module()
        self.assertEqual(module.ID, "BO01")
        self.assertEqual(module.FAMILY_ID, "LBC")
        self.assertEqual(module.NAME, "BO01Strategy")
        self.assertEqual(module.EXPLICIT_TIMEFRAME, "M5")
        self.assertIsInstance(module.WARMUP_BARS, int)
        self.assertIsInstance(module.DEFAULT_PARAMS, dict)
        self.assertTrue(callable(module.default_params))
        self.assertTrue(callable(module.parameter_space))
        self.assertTrue(callable(module.parameter_grid))
        self.assertEqual(list(inspect.signature(module.signal).parameters), ["frame", "i", "params"])

    def test_signal_contract_and_no_file_access_during_signal(self) -> None:
        module = _module()
        frame, i = _bo01_frame()
        with patch.object(builtins, "open", side_effect=AssertionError("file access")):
            with patch.object(pd, "read_" + "csv", side_effect=AssertionError("tabular file access")):
                result = module.signal(frame, i, module.default_params())
        self.assertEqual(result["signal"], 1)
        self.assertEqual(result["direction"], "long")
        _assert_signal_contract(self, result)

    def test_small_frame_fails_closed(self) -> None:
        module = _module()
        frame, i = _bo01_frame()
        tiny = frame.iloc[:5].copy()
        self.assertIsNone(module.signal(tiny, min(2, len(tiny) - 1), module.default_params()))

    def test_no_future_poisoning_changes_current_signal(self) -> None:
        module = _module()
        frame, i = _bo01_frame()
        baseline = module.signal(frame, i, module.default_params())
        poisoned = frame.copy()
        cols = [poisoned.columns.get_loc(c) for c in ("open", "high", "low", "close", "volume", "ema_m15_200")]
        poisoned.iloc[i + 1 :, cols] = 9.9999
        mutated = module.signal(poisoned, i, module.default_params())
        self.assertEqual(baseline, mutated)

    def test_current_bar_boundary_and_warmup_gate(self) -> None:
        module = _module()
        frame, _ = _bo01_frame()
        i = module.WARMUP_BARS - 1
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_fail_closed_missing_columns_tz_naive_nan_and_state(self) -> None:
        module = _module()
        frame, i = _bo01_frame()
        self.assertIsNone(module.signal(frame.drop(columns=["ema_m15_200"]), i, module.default_params()))
        naive = frame.copy()
        naive.index = naive.index.tz_localize(None)
        self.assertIsNone(module.signal(naive, i, module.default_params()))
        nan_frame = frame.copy()
        nan_frame.iloc[i, nan_frame.columns.get_loc("close")] = np.nan
        self.assertIsNone(module.signal(nan_frame, i, module.default_params()))
        params = module.default_params()
        params["daily_trade_count"] = 1
        self.assertIsNone(module.signal(frame, i, params))
        params = module.default_params()
        params["has_active_position"] = True
        self.assertIsNone(module.signal(frame, i, params))

    def test_forbidden_scope_tokens_absent_from_skeleton_source(self) -> None:
        source = inspect.getsource(_module())
        blocked = [
            "20" + "25",
            "20" + "26",
            "hold" + "out",
            "sealed_" + "hold" + "out",
            "valid" + "ation",
            "formal_train_" + "runner",
            "--" + "execute",
            "micro-" + "run",
            "dry-" + "run",
            "back" + "test",
            "optim" + "ization",
            "swe" + "ep",
        ]
        for token in blocked:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
