"""Timezone / index / cadence contract guards.

Encodes the time-contract findings from
``ENGINE_STRATEGY_CONTRACT_AUDIT_VEORB_V1.md``:

  * C3 — the strategy frame index is timezone-aware and resolves to NY local
    wall-clock (loader: ``parse_prepared_index -> tz_convert(NY_TZ)``; engine
    independently re-derives NY via ``tz_convert``). DST is handled by
    ``entry_open_index`` (EST -05:00 vs EDT -04:00).
  * C8 — ``_infer_cadence_minutes`` is robust to weekend/holiday gaps for a
    5-minute series (stays 5) BUT silently returns ``None`` on irregular
    cadence, which would suppress ALL signals with no engine-level alarm.
    This is a documented fragility, encoded here so it cannot regress
    unnoticed.

Lightweight only: tiny synthetic frames, no market data, no backtest sweep.
"""
from __future__ import annotations

import types
import unittest

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig, NY_TZ
from research_lab.engine import entry_open_index, run_backtest
from research_lab.strategies import ve_orb_volatility_expansion as veorb


class OneShotLong:
    NAME = "one_shot_long_tz"
    WARMUP_BARS = 0

    @staticmethod
    def signal(frame, i, params):
        if frame.index[i].strftime("%H:%M") == "11:00":
            return {"direction": "long", "stop_mode": "atr", "stop_atr": 1.0,
                    "target_rr": 1.0, "session_name": "all_day"}
        return None


def _frame(rows: list[dict], *, tz) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(r["timestamp"], tz=tz) for r in rows])
    return pd.DataFrame(
        {
            "open": [r["open"] for r in rows], "high": [r["high"] for r in rows],
            "low": [r["low"] for r in rows], "close": [r["close"] for r in rows],
            "atr14": [r.get("atr14", 0.0010) for r in rows],
            "range_atr": [r.get("range_atr", 0.5) for r in rows],
        },
        index=idx,
    )


_BARS = [
    {"timestamp": "2022-01-03 10:45:00", "open": 1.1000, "high": 1.1004, "low": 1.0998, "close": 1.1002},
    {"timestamp": "2022-01-03 11:00:00", "open": 1.1002, "high": 1.1006, "low": 1.1000, "close": 1.1004},
    {"timestamp": "2022-01-03 11:15:00", "open": 1.1004, "high": 1.1018, "low": 1.1003, "close": 1.1015},
    {"timestamp": "2022-01-03 11:30:00", "open": 1.1015, "high": 1.1016, "low": 1.1010, "close": 1.1012},
]


class TimezoneIndexContractTests(unittest.TestCase):
    """TEST GROUP E — timezone / index."""

    def test_entry_open_index_respects_dst_offsets(self):
        """EST (-05:00) vs EDT (-04:00): the session anchor must follow DST."""
        edt = entry_open_index(pd.DatetimeIndex([
            pd.Timestamp("2022-07-05 11:00:00", tz=NY_TZ),
            pd.Timestamp("2022-07-05 11:15:00", tz=NY_TZ),
        ]))
        est = entry_open_index(pd.DatetimeIndex([
            pd.Timestamp("2022-01-05 11:00:00", tz=NY_TZ),
            pd.Timestamp("2022-01-05 11:15:00", tz=NY_TZ),
        ]))
        self.assertEqual(edt[1].strftime("%z"), "-0400")
        self.assertEqual(est[1].strftime("%z"), "-0500")

    def test_strategy_receives_tz_aware_ny_index(self):
        """Contract: the frame handed to signal() is tz-aware NY wall-clock."""
        seen: list = []
        spy = types.SimpleNamespace(
            NAME="tz_spy", WARMUP_BARS=0,
            signal=lambda f, i, p: seen.append((f.index.tz, f.index[i])) or None,
        )
        frame = _frame(_BARS, tz=NY_TZ)
        run_backtest(spy, frame, {}, EngineConfig(pair="EURUSD", max_trades_per_day=2),
                     np.zeros(len(frame), dtype=bool), False)
        self.assertTrue(seen, "signal() never called")
        tz0, ts0 = seen[0]
        self.assertIsNotNone(tz0, "strategy received a tz-naive index")
        self.assertEqual(str(tz0), NY_TZ)
        self.assertEqual(ts0.tz_convert(NY_TZ).strftime("%H:%M"),
                         ts0.strftime("%H:%M"), "index is not NY wall-clock")

    def test_engine_localizes_naive_index_as_utc_then_ny(self):
        """engine.py:636 contract: a tz-naive index is treated as UTC then
        converted to NY (must not crash, must still trade)."""
        frame = _frame(_BARS, tz=None)
        self.assertIsNone(frame.index.tz)
        result = run_backtest(
            OneShotLong, frame, {},
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.0,
                         max_spread_pips=2.0, slippage_pips=0.0,
                         commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool), False,
        )
        self.assertFalse(result.equity_curve.empty,
                         "engine produced no equity on a tz-naive frame")


class CadenceInferenceFragilityTests(unittest.TestCase):
    """TEST GROUP E — cadence inference (documented C8 fragility)."""

    def test_m5_with_weekend_gaps_infers_stable_cadence_5(self):
        idx = []
        base = pd.Timestamp("2020-01-06 00:00:00", tz=NY_TZ)  # Monday
        for day in range(5):              # Mon..Fri, 60 bars/day, strict 5-min
            for b in range(60):
                idx.append(base + pd.Timedelta(days=day) + pd.Timedelta(minutes=5 * b))
        idx.append(base + pd.Timedelta(days=7))  # jump over the weekend gap
        cadence = veorb._infer_cadence_minutes(pd.DatetimeIndex(idx))
        self.assertEqual(cadence, 5, "M5 + weekend gaps must still infer cadence=5")

    def test_irregular_cadence_silently_returns_none(self):
        """Documents the silent-kill fragility: an unsupported cadence yields
        None -> _opening_range -> no signal, with NO engine-level alarm."""
        irregular = pd.DatetimeIndex([
            pd.Timestamp("2020-01-06 00:00:00", tz=NY_TZ),
            pd.Timestamp("2020-01-06 00:07:00", tz=NY_TZ),
            pd.Timestamp("2020-01-06 00:18:00", tz=NY_TZ),
            pd.Timestamp("2020-01-06 00:31:00", tz=NY_TZ),
        ])
        self.assertIsNone(veorb._infer_cadence_minutes(irregular),
                          "non-standard cadence must be the documented None case")


if __name__ == "__main__":
    unittest.main()
