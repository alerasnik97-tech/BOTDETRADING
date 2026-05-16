from __future__ import annotations

import builtins
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_lab.config import NY_TZ
from research_lab.strategies import STRATEGY_REGISTRY
from research_lab.strategies import (
    mr01_anchor_elastic,
    mr02_vwap_stretch_reversion,
    tp01_london_ny_momentum_pullback,
    ve_orb_volatility_expansion,
)


APPROVED_MODULES = (
    mr01_anchor_elastic,
    mr02_vwap_stretch_reversion,
    tp01_london_ny_momentum_pullback,
    ve_orb_volatility_expansion,
)
APPROVED_KEYS = (
    "mr01_anchor_elastic",
    "mr02_vwap_stretch_reversion",
    "tp01_london_ny_momentum_pullback",
    "ve_orb_volatility_expansion",
)


def _signal_value(result: dict | None) -> int:
    if result is None:
        return 0
    return int(result["signal"])


def _assert_engine_contract(testcase: unittest.TestCase, result: dict | None) -> None:
    if result is None:
        return
    testcase.assertIn(result["signal"], {-1, 1})
    testcase.assertIn(result["direction"], {"long", "short"})
    testcase.assertEqual(result["stop_mode"], "price")
    testcase.assertIn("stop_price", result)
    testcase.assertIn(result["target_mode"], {"price", "rr"})


def _source_text() -> str:
    return "\n".join(inspect.getsource(module) for module in APPROVED_MODULES).lower()


def _minute_frame(start: str, periods: int, base: float = 1.1000) -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(start, tz=NY_TZ), periods=periods, freq="min")
    close = np.full(periods, base, dtype=float)
    open_ = close.copy()
    high = close + 0.00010
    low = close - 0.00010
    volume = np.full(periods, 1000.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr14": np.full(periods, 0.00050),
            "adx14": np.full(periods, 12.0),
        },
        index=index,
    )


def _mr_frame(module_name: str, direction: str) -> tuple[pd.DataFrame, int]:
    frame = _minute_frame("2024-01-02 07:00:00", 34)
    for idx in range(30):
        frame.iloc[idx, frame.columns.get_loc("close")] = 1.1000 + (0.00003 if idx % 2 else -0.00003)
        frame.iloc[idx, frame.columns.get_loc("open")] = frame["close"].iat[idx]
        frame.iloc[idx, frame.columns.get_loc("high")] = frame["close"].iat[idx] + 0.00010
        frame.iloc[idx, frame.columns.get_loc("low")] = frame["close"].iat[idx] - 0.00010
    i = 31
    if direction == "long":
        extreme = 1.0968 if module_name == "mr02" else 1.0980
        reentry = 1.0990 if module_name == "mr02" else 1.0994
        frame.iloc[i - 1, frame.columns.get_loc("close")] = extreme
        frame.iloc[i - 1, frame.columns.get_loc("low")] = extreme - 0.00020
        frame.iloc[i - 1, frame.columns.get_loc("high")] = extreme + 0.00010
        frame.iloc[i, frame.columns.get_loc("close")] = reentry
        frame.iloc[i, frame.columns.get_loc("low")] = reentry - 0.00015
        frame.iloc[i, frame.columns.get_loc("high")] = reentry + 0.00010
    else:
        extreme = 1.1032 if module_name == "mr02" else 1.1020
        reentry = 1.1010 if module_name == "mr02" else 1.1006
        frame.iloc[i - 1, frame.columns.get_loc("close")] = extreme
        frame.iloc[i - 1, frame.columns.get_loc("high")] = extreme + 0.00020
        frame.iloc[i - 1, frame.columns.get_loc("low")] = extreme - 0.00010
        frame.iloc[i, frame.columns.get_loc("close")] = reentry
        frame.iloc[i, frame.columns.get_loc("high")] = reentry + 0.00015
        frame.iloc[i, frame.columns.get_loc("low")] = reentry - 0.00010
    return frame, i


def _tp_frame(*, active: bool = True, direction: str = "long") -> tuple[pd.DataFrame, int]:
    periods = 270
    frame = _minute_frame("2024-01-02 04:00:00", periods, base=1.1000)
    close = np.full(periods, 1.1000)
    if active:
        i = 260
        if direction == "long":
            for idx in range(periods):
                close[idx] = 1.1000 + idx * 0.000002
            close[i - 6] = 1.1000
            close[i - 5] = 1.1002
            close[i - 4] = 1.1004
            close[i - 3] = 1.1006
            close[i - 2] = 1.1008
            close[i - 1] = 1.1009
            close[i] = 1.1011
        else:
            for idx in range(periods):
                close[idx] = 1.1015 - idx * 0.000002
            close[i - 6] = 1.1012
            close[i - 5] = 1.1010
            close[i - 4] = 1.1008
            close[i - 3] = 1.1006
            close[i - 2] = 1.1004
            close[i - 1] = 1.1003
            close[i] = 1.1001
    else:
        i = 260
        close[:] = 1.1000 + np.sin(np.arange(periods) / 5.0) * 0.00002
    frame["close"] = close
    frame["open"] = close
    frame["high"] = close + 0.00010
    frame["low"] = close - 0.00010
    frame.loc[frame.index[i], "high"] = frame["close"].iat[i] + 0.00110
    frame.loc[frame.index[i], "low"] = frame["close"].iat[i] - 0.00055
    if active:
        if direction == "long":
            frame.loc[frame.index[i - 1], "high"] = frame["close"].iat[i - 1] + 0.00005
            frame.loc[frame.index[i], "low"] = frame["close"].iat[i] - 0.00070
        else:
            frame.loc[frame.index[i - 1], "low"] = frame["close"].iat[i - 1] - 0.00005
            frame.loc[frame.index[i], "high"] = frame["close"].iat[i] + 0.00070
    return frame, i


def _orb_frame(*, direction: str = "long") -> tuple[pd.DataFrame, int, int]:
    periods = 280
    frame = _minute_frame("2024-01-02 04:00:00", periods, base=1.1000)
    frame["high"] = 1.10010
    frame["low"] = 1.09990
    frame["close"] = 1.10000
    or_mask = (frame.index.hour == 7)
    frame.loc[or_mask, "high"] = 1.10030
    frame.loc[or_mask, "low"] = 1.09970
    frame.loc[or_mask, "close"] = 1.10000
    pre_i = int(frame.index.get_loc(pd.Timestamp("2024-01-02 07:30:00", tz=NY_TZ)))
    i = int(frame.index.get_loc(pd.Timestamp("2024-01-02 08:20:00", tz=NY_TZ)))
    if direction == "long":
        frame.iloc[i - 1, frame.columns.get_loc("close")] = 1.10020
        frame.iloc[i, frame.columns.get_loc("close")] = 1.10055
        frame.iloc[i, frame.columns.get_loc("high")] = 1.10140
        frame.iloc[i, frame.columns.get_loc("low")] = 1.09995
    else:
        frame.iloc[i - 1, frame.columns.get_loc("close")] = 1.09980
        frame.iloc[i, frame.columns.get_loc("close")] = 1.09945
        frame.iloc[i, frame.columns.get_loc("high")] = 1.10005
        frame.iloc[i, frame.columns.get_loc("low")] = 1.09860
    return frame, i, pre_i


def _orb_incomplete_frame() -> tuple[pd.DataFrame, int]:
    index = list(pd.date_range(pd.Timestamp("2024-01-02 03:20:00", tz=NY_TZ), pd.Timestamp("2024-01-02 06:59:00", tz=NY_TZ), freq="min"))
    index.append(pd.Timestamp("2024-01-02 07:00:00", tz=NY_TZ))
    index.extend(pd.date_range(pd.Timestamp("2024-01-02 08:00:00", tz=NY_TZ), pd.Timestamp("2024-01-02 08:20:00", tz=NY_TZ), freq="min"))
    frame = pd.DataFrame(index=pd.DatetimeIndex(index))
    frame["open"] = 1.1000
    frame["high"] = 1.1001
    frame["low"] = 1.0999
    frame["close"] = 1.1000
    frame["volume"] = 1000.0
    frame["atr14"] = 0.00050
    frame["adx14"] = 12.0
    frame.loc[pd.Timestamp("2024-01-02 07:00:00", tz=NY_TZ), ["high", "low", "close"]] = [1.1003, 1.0997, 1.1000]
    frame.loc[pd.Timestamp("2024-01-02 08:19:00", tz=NY_TZ), "close"] = 1.1002
    frame.loc[pd.Timestamp("2024-01-02 08:20:00", tz=NY_TZ), ["high", "low", "close"]] = [1.1014, 1.09995, 1.10055]
    i = int(frame.index.get_loc(pd.Timestamp("2024-01-02 08:20:00", tz=NY_TZ)))
    return frame, i


class PriorityASkeletonTests(unittest.TestCase):
    def test_imports_and_signal_values_are_contractual(self) -> None:
        for module in APPROVED_MODULES:
            self.assertTrue(hasattr(module, "signal"))
            self.assertEqual(_signal_value(module.signal(_minute_frame("2024-01-02 07:00:00", 4), 1, module.default_params())), 0)

    def test_fail_closed_with_insufficient_data(self) -> None:
        tiny = _minute_frame("2024-01-02 07:00:00", 3)
        for module in APPROVED_MODULES:
            self.assertIsNone(module.signal(tiny, 2, module.default_params()))

    def test_no_file_access_during_signal_calls(self) -> None:
        cases = [
            (mr01_anchor_elastic, *_mr_frame("mr01", "long")),
            (mr02_vwap_stretch_reversion, *_mr_frame("mr02", "long")),
            (tp01_london_ny_momentum_pullback, *_tp_frame(active=True)),
            (ve_orb_volatility_expansion, *_orb_frame()[:2]),
        ]
        read_name = "read_" + "csv"
        with patch.object(builtins, "open", side_effect=AssertionError("file access")):
            with patch.object(pd, read_name, side_effect=AssertionError("tabular file access")):
                for module, frame, i in cases:
                    result = module.signal(frame, i, module.default_params())
                    _assert_engine_contract(self, result)

    def test_source_has_no_blocked_external_dependencies(self) -> None:
        text = _source_text()
        blocked = [
            "20" + "25",
            "20" + "26",
            "hold" + "out",
            "sealed_" + "hold" + "out",
            "forex_" + "factory",
            "ne" + "ws",
            "high" + "_precision",
            "level" + "2",
            "clean-" + "sync",
            "z" + "ip",
            "read_" + "csv",
            "to_" + "csv",
            "o" + "pen(",
            "P" + "ath(",
        ]
        for token in blocked:
            self.assertNotIn(token.lower(), text)

    def test_registry_contains_only_approved_new_keys(self) -> None:
        for key in APPROVED_KEYS:
            self.assertIn(key, STRATEGY_REGISTRY)
        excluded = ["ve" + "01", "sd" + "01", "ed" + "01"]
        for token in excluded:
            self.assertNotIn(token, APPROVED_KEYS)

    def test_mr01_emits_long_and_short_on_synthetic_extremes(self) -> None:
        long_frame, long_i = _mr_frame("mr01", "long")
        short_frame, short_i = _mr_frame("mr01", "short")
        long_signal = mr01_anchor_elastic.signal(long_frame, long_i, mr01_anchor_elastic.default_params())
        short_signal = mr01_anchor_elastic.signal(short_frame, short_i, mr01_anchor_elastic.default_params())
        self.assertEqual(_signal_value(long_signal), 1)
        self.assertEqual(_signal_value(short_signal), -1)
        _assert_engine_contract(self, long_signal)
        _assert_engine_contract(self, short_signal)

    def test_mr02_emits_long_and_short_on_synthetic_band_reentry(self) -> None:
        long_frame, long_i = _mr_frame("mr02", "long")
        short_frame, short_i = _mr_frame("mr02", "short")
        long_signal = mr02_vwap_stretch_reversion.signal(long_frame, long_i, mr02_vwap_stretch_reversion.default_params())
        short_signal = mr02_vwap_stretch_reversion.signal(short_frame, short_i, mr02_vwap_stretch_reversion.default_params())
        self.assertEqual(_signal_value(long_signal), 1)
        self.assertEqual(_signal_value(short_signal), -1)
        _assert_engine_contract(self, long_signal)
        _assert_engine_contract(self, short_signal)

    def test_tp01_requires_momentum_pullback_not_lateral_range(self) -> None:
        active_frame, active_i = _tp_frame(active=True)
        lateral_frame, lateral_i = _tp_frame(active=False)
        active_signal = tp01_london_ny_momentum_pullback.signal(
            active_frame,
            active_i,
            tp01_london_ny_momentum_pullback.default_params(),
        )
        lateral_signal = tp01_london_ny_momentum_pullback.signal(
            lateral_frame,
            lateral_i,
            tp01_london_ny_momentum_pullback.default_params(),
        )
        self.assertEqual(_signal_value(active_signal), 1)
        self.assertIsNone(lateral_signal)
        _assert_engine_contract(self, active_signal)

    def test_tp01_short_signal(self) -> None:
        frame, i = _tp_frame(active=True, direction="short")
        result = tp01_london_ny_momentum_pullback.signal(
            frame,
            i,
            tp01_london_ny_momentum_pullback.default_params(),
        )
        self.assertEqual(_signal_value(result), -1)
        _assert_engine_contract(self, result)

    def test_ve_orb_fails_closed_with_incomplete_opening_range(self) -> None:
        frame, i = _orb_incomplete_frame()
        result = ve_orb_volatility_expansion.signal(frame, i, ve_orb_volatility_expansion.default_params())
        self.assertIsNone(result)

    def test_ve_orb_allows_complete_opening_range(self) -> None:
        frame, i, pre_i = _orb_frame()
        params = ve_orb_volatility_expansion.default_params()
        self.assertIsNone(ve_orb_volatility_expansion.signal(frame, pre_i, params))
        result = ve_orb_volatility_expansion.signal(frame, i, params)
        self.assertEqual(_signal_value(result), 1)
        _assert_engine_contract(self, result)

    def test_ve_orb_short_signal(self) -> None:
        frame, i, _ = _orb_frame(direction="short")
        result = ve_orb_volatility_expansion.signal(frame, i, ve_orb_volatility_expansion.default_params())
        self.assertEqual(_signal_value(result), -1)
        _assert_engine_contract(self, result)

    def test_nan_critical_inputs_fail_closed(self) -> None:
        frame, i = _mr_frame("mr01", "long")
        frame.iloc[i, frame.columns.get_loc("close")] = np.nan
        self.assertIsNone(mr01_anchor_elastic.signal(frame, i, mr01_anchor_elastic.default_params()))
        frame2, i2, _ = _orb_frame()
        frame2.iloc[i2, frame2.columns.get_loc("high")] = np.nan
        self.assertIsNone(ve_orb_volatility_expansion.signal(frame2, i2, ve_orb_volatility_expansion.default_params()))

    def test_mr02_nan_fail_closed(self) -> None:
        for column in ("close", "high", "low"):
            frame, i = _mr_frame("mr02", "long")
            frame.iloc[i, frame.columns.get_loc(column)] = np.nan
            self.assertIsNone(mr02_vwap_stretch_reversion.signal(frame, i, mr02_vwap_stretch_reversion.default_params()))
        frame, i = _mr_frame("mr02", "long")
        frame.iloc[i - 5, frame.columns.get_loc("volume")] = np.nan
        self.assertIsNone(mr02_vwap_stretch_reversion.signal(frame, i, mr02_vwap_stretch_reversion.default_params()))

    def test_tp01_nan_fail_closed(self) -> None:
        for column in ("close", "high", "low"):
            frame, i = _tp_frame(active=True)
            frame.iloc[i, frame.columns.get_loc(column)] = np.nan
            self.assertIsNone(tp01_london_ny_momentum_pullback.signal(frame, i, tp01_london_ny_momentum_pullback.default_params()))

    def test_no_disallowed_family_tokens_in_sources(self) -> None:
        text = _source_text()
        blocked = ["r" + "v5", "r" + "v15", "p" + "30", "0." + "08", "europe_extreme", "post_stabilization"]
        for token in blocked:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
