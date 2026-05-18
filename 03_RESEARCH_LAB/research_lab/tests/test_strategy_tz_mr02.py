from __future__ import annotations

import importlib
import unittest

import pandas as pd


MODULE_PATH = "research_lab.strategies.MR02Strategy"


def _module():
    return importlib.import_module(MODULE_PATH)


def _target_index(frame: pd.DataFrame, date: str, hhmm: str) -> int:
    target = pd.Timestamp(f"{date} {hhmm}:00", tz="UTC").tz_convert(frame.index.tz)
    return int(frame.index.get_loc(target))


def _replace_utc_timestamp(frame: pd.DataFrame, date: str, old_hhmm: str, new_hhmm: str) -> pd.DataFrame:
    modified = frame.copy()
    old_stamp = pd.Timestamp(f"{date} {old_hhmm}:00", tz="UTC").tz_convert(modified.index.tz)
    new_stamp = pd.Timestamp(f"{date} {new_hhmm}:00", tz="UTC").tz_convert(modified.index.tz)
    index_values = list(modified.index)
    index_values[index_values.index(old_stamp)] = new_stamp
    modified.index = pd.DatetimeIndex(index_values)
    return modified


def _mr02_frame(date: str, hhmm: str, *, tz: str = "UTC", active: bool = True) -> tuple[pd.DataFrame, int]:
    index_utc = pd.date_range(pd.Timestamp(f"{date} 00:00:00", tz="UTC"), periods=160, freq="5min")
    index = index_utc.tz_convert(tz)
    frame = pd.DataFrame(index=index)
    frame["open"] = 1.1000
    frame["high"] = 1.1004
    frame["low"] = 1.0994
    frame["close"] = 1.1000
    frame["atr14"] = 0.0010
    i = _target_index(frame, date, hhmm)
    prior = i - 1
    if active:
        frame.iloc[prior, frame.columns.get_loc("open")] = 1.10050
        frame.iloc[prior, frame.columns.get_loc("close")] = 1.10065
        frame.iloc[prior, frame.columns.get_loc("high")] = 1.10075
        frame.iloc[prior, frame.columns.get_loc("low")] = 1.10045
        frame.iloc[i, frame.columns.get_loc("open")] = 1.10070
        frame.iloc[i, frame.columns.get_loc("close")] = 1.10035
        frame.iloc[i, frame.columns.get_loc("high")] = 1.10072
        frame.iloc[i, frame.columns.get_loc("low")] = 1.10030
    return frame, i


class MR02TimezoneTests(unittest.TestCase):
    def test_gmt_session_accepts_0710_utc(self) -> None:
        module = _module()
        frame, i = _mr02_frame("2024-03-12", "07:10")
        result = module.signal(frame, i, module.default_params())
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "short")

    def test_dst_march_uses_gmt_not_local_hour(self) -> None:
        module = _module()
        frame, i = _mr02_frame("2024-03-12", "07:10", tz="America/New_York")
        self.assertNotEqual(frame.index[i].hour, 7)
        result = module.signal(frame, i, module.default_params())
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], -1)

    def test_dst_november_uses_gmt_not_local_hour(self) -> None:
        module = _module()
        frame, i = _mr02_frame("2024-11-05", "07:10", tz="America/New_York")
        self.assertNotEqual(frame.index[i].hour, 7)
        result = module.signal(frame, i, module.default_params())
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], -1)

    def test_no_signal_before_entry_window(self) -> None:
        module = _module()
        frame, i = _mr02_frame("2024-03-12", "06:55")
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_no_signal_when_asian_endpoint_0630_missing(self) -> None:
        module = _module()
        frame, i = _mr02_frame("2024-03-12", "07:10")
        frame = _replace_utc_timestamp(frame, "2024-03-12", "06:30", "06:29")
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_no_signal_after_entry_window(self) -> None:
        module = _module()
        frame, i = _mr02_frame("2024-03-12", "11:05")
        self.assertIsNone(module.signal(frame, i, module.default_params()))

    def test_no_entry_without_objective_break_and_failure(self) -> None:
        module = _module()
        frame, i = _mr02_frame("2024-03-12", "07:10", active=False)
        self.assertIsNone(module.signal(frame, i, module.default_params()))


if __name__ == "__main__":
    unittest.main()
