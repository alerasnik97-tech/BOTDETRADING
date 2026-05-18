from __future__ import annotations

import builtins
import importlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_PATH = "research_lab.strategies.MR02Strategy"


def _module():
    return importlib.import_module(MODULE_PATH)


def _target_index(frame: pd.DataFrame, date: str, hhmm: str) -> int:
    target = pd.Timestamp(f"{date} {hhmm}:00", tz="UTC").tz_convert(frame.index.tz)
    return int(frame.index.get_loc(target))


def _utc_stamp(frame: pd.DataFrame, date: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{date} {hhmm}:00", tz="UTC").tz_convert(frame.index.tz)


def _replace_utc_timestamp(frame: pd.DataFrame, date: str, old_hhmm: str, new_hhmm: str) -> pd.DataFrame:
    modified = frame.copy()
    old_stamp = _utc_stamp(modified, date, old_hhmm)
    new_stamp = _utc_stamp(modified, date, new_hhmm)
    index_values = list(modified.index)
    index_values[index_values.index(old_stamp)] = new_stamp
    modified.index = pd.DatetimeIndex(index_values)
    return modified


def _mr02_frame(date: str = "2024-03-12", *, hhmm: str = "07:10", tz: str = "UTC",
                direction: str = "short") -> tuple[pd.DataFrame, int]:
    index_utc = pd.date_range(pd.Timestamp(f"{date} 00:00:00", tz="UTC"), periods=155, freq="5min")
    index = index_utc.tz_convert(tz)
    frame = pd.DataFrame(index=index)
    frame["open"] = 1.1000
    frame["high"] = 1.1004
    frame["low"] = 1.0994
    frame["close"] = 1.1000
    frame["atr14"] = 0.0010
    i = _target_index(frame, date, hhmm)
    prior = i - 1
    if direction == "short":
        frame.iloc[prior, frame.columns.get_loc("open")] = 1.10050
        frame.iloc[prior, frame.columns.get_loc("close")] = 1.10065
        frame.iloc[prior, frame.columns.get_loc("high")] = 1.10075
        frame.iloc[prior, frame.columns.get_loc("low")] = 1.10045
        frame.iloc[i, frame.columns.get_loc("open")] = 1.10070
        frame.iloc[i, frame.columns.get_loc("close")] = 1.10035
        frame.iloc[i, frame.columns.get_loc("high")] = 1.10072
        frame.iloc[i, frame.columns.get_loc("low")] = 1.10030
    else:
        frame.iloc[prior, frame.columns.get_loc("open")] = 1.09930
        frame.iloc[prior, frame.columns.get_loc("close")] = 1.09915
        frame.iloc[prior, frame.columns.get_loc("high")] = 1.09935
        frame.iloc[prior, frame.columns.get_loc("low")] = 1.09905
        frame.iloc[i, frame.columns.get_loc("open")] = 1.09910
        frame.iloc[i, frame.columns.get_loc("close")] = 1.09945
        frame.iloc[i, frame.columns.get_loc("high")] = 1.09950
        frame.iloc[i, frame.columns.get_loc("low")] = 1.09908
    return frame, i


def _assert_signal_contract(testcase: unittest.TestCase, result: dict | None) -> None:
    if result is None:
        return
    testcase.assertIn(result["signal"], {-1, 1})
    testcase.assertIn(result["direction"], {"long", "short"})
    testcase.assertEqual(result["stop_mode"], "price")
    testcase.assertIn("stop_price", result)
    testcase.assertEqual(result["target_mode"], "rr")
    testcase.assertEqual(result["target_rr"], 1.5)
    testcase.assertIn("session_name", result)


class MR02ContractTests(unittest.TestCase):
    def test_import_and_module_contract(self) -> None:
        module = _module()
        self.assertEqual(module.ID, "MR02")
        self.assertEqual(module.FAMILY_ID, "LBF")
        self.assertEqual(module.NAME, "MR02Strategy")
        self.assertEqual(module.EXPLICIT_TIMEFRAME, "M5")
        self.assertIsInstance(module.WARMUP_BARS, int)
        self.assertIsInstance(module.DEFAULT_PARAMS, dict)
        self.assertTrue(callable(module.default_params))
        self.assertTrue(callable(module.parameter_space))
        self.assertTrue(callable(module.parameter_grid))
        self.assertEqual(list(inspect.signature(module.signal).parameters), ["frame", "i", "params"])

    def test_signal_contract_and_no_file_access_during_signal(self) -> None:
        module = _module()
        frame, i = _mr02_frame()
        with patch.object(builtins, "open", side_effect=AssertionError("file access")):
            with patch.object(pd, "read_" + "csv", side_effect=AssertionError("tabular file access")):
                result = module.signal(frame, i, module.default_params())
        self.assertEqual(result["signal"], -1)
        self.assertEqual(result["direction"], "short")
        _assert_signal_contract(self, result)

    def test_missing_asian_endpoint_0630_fails_closed(self) -> None:
        module = _module()
        frame, i = _mr02_frame()
        frame = _replace_utc_timestamp(frame, "2024-03-12", "06:30", "06:29")
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_duplicate_asian_timestamp_replacing_missing_bar_fails_closed(self) -> None:
        module = _module()
        frame, i = _mr02_frame()
        frame = _replace_utc_timestamp(frame, "2024-03-12", "06:25", "06:20")
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_wrong_asian_cadence_fails_closed(self) -> None:
        module = _module()
        frame, i = _mr02_frame()
        frame = _replace_utc_timestamp(frame, "2024-03-12", "00:05", "00:07")
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_long_side_eligible_fakeout_signal_contract(self) -> None:
        module = _module()
        frame, i = _mr02_frame(direction="long")
        result = module.signal(frame, i, module.default_params())
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], 1)
        self.assertEqual(result["direction"], "long")
        self.assertEqual(result["target_rr"], 1.5)
        self.assertLess(result["stop_price"], float(frame["close"].iat[i]))
        _assert_signal_contract(self, result)

    def test_third_prior_bar_breach_remains_eligible(self) -> None:
        module = _module()
        frame, i = _mr02_frame()
        columns = {name: frame.columns.get_loc(name) for name in ("open", "high", "low", "close")}
        frame.iloc[i - 3, columns["open"]] = 1.10020
        frame.iloc[i - 3, columns["high"]] = 1.10075
        frame.iloc[i - 3, columns["low"]] = 1.10010
        frame.iloc[i - 3, columns["close"]] = 1.10030
        frame.iloc[i - 2, columns["open"]] = 1.10005
        frame.iloc[i - 2, columns["high"]] = 1.10030
        frame.iloc[i - 2, columns["low"]] = 1.10000
        frame.iloc[i - 2, columns["close"]] = 1.10010
        frame.iloc[i - 1, columns["open"]] = 1.10010
        frame.iloc[i - 1, columns["high"]] = 1.10035
        frame.iloc[i - 1, columns["low"]] = 1.10005
        frame.iloc[i - 1, columns["close"]] = 1.10020
        frame.iloc[i, columns["open"]] = 1.10025
        frame.iloc[i, columns["high"]] = 1.10030
        frame.iloc[i, columns["low"]] = 1.10000
        frame.iloc[i, columns["close"]] = 1.10005
        result = module.signal(frame, i, module.default_params())
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], -1)
        self.assertEqual(result["direction"], "short")

    def test_small_frame_fails_closed(self) -> None:
        module = _module()
        frame, _ = _mr02_frame()
        tiny = frame.iloc[:5].copy()
        self.assertIsNone(module.signal(tiny, min(2, len(tiny) - 1), module.default_params()))

    def test_no_future_poisoning_changes_current_signal(self) -> None:
        module = _module()
        frame, i = _mr02_frame()
        baseline = module.signal(frame, i, module.default_params())
        poisoned = frame.copy()
        cols = [poisoned.columns.get_loc(c) for c in ("open", "high", "low", "close")]
        poisoned.iloc[i + 1 :, cols] = 9.9999
        mutated = module.signal(poisoned, i, module.default_params())
        self.assertEqual(baseline, mutated)

    def test_current_bar_boundary_and_warmup_gate(self) -> None:
        module = _module()
        frame, _ = _mr02_frame()
        i = module.WARMUP_BARS - 1
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_fail_closed_missing_columns_tz_naive_nan_and_state(self) -> None:
        module = _module()
        frame, i = _mr02_frame()
        self.assertIsNone(module.signal(frame.drop(columns=["high"]), i, module.default_params()))
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
