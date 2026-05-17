"""Engine <-> strategy contract guards + universal anti-lookahead harness.

Encodes, as executable tests, the contract documented in
``06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/ENGINE_STRATEGY_CONTRACT_AUDIT_VEORB_V1.md``:

  * The engine invokes ``strategy_module.signal(frame, i, params)`` (or
    ``generate_signal`` when present) passing the **entire multi-year frame**
    plus the integer bar index ``i``. There is NO causal sandbox: a strategy
    must restrict itself to rows ``<= i``.
  * Entry fill is **T+1** (signal observed at bar ``i`` -> fill on the next
    bar's open), so the execution path itself never leaks the future.

These tests FAIL if the contract drifts (e.g. the engine starts slicing the
frame, fills same-bar, or the documented signature changes silently) and the
anti-lookahead harness FAILS if a strategy reads rows ``> i``.

Lightweight only: tiny synthetic in-memory frames, no market data, no backtest
sweep, no validation/holdout.
"""
from __future__ import annotations

import inspect
import types
import unittest

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig, NY_TZ
from research_lab.engine import run_backtest
from research_lab.strategies import ve_orb_volatility_expansion as veorb
from research_lab.strategies import tp01_london_ny_momentum_pullback as tp01


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _ny_ts(value) -> pd.Timestamp:
    """NY-localized Timestamp, robust to naive strings AND tz-aware inputs."""
    t = pd.Timestamp(value)
    return t.tz_localize(NY_TZ) if t.tzinfo is None else t.tz_convert(NY_TZ)


def make_ohlc_frame(rows: list[dict]) -> pd.DataFrame:
    """Engine-compatible frame (NY tz-aware index, atr14/range_atr columns)."""
    index = pd.DatetimeIndex([_ny_ts(r["timestamp"]) for r in rows])
    return pd.DataFrame(
        {
            "open": [r["open"] for r in rows],
            "high": [r["high"] for r in rows],
            "low": [r["low"] for r in rows],
            "close": [r["close"] for r in rows],
            "atr14": [r.get("atr14", 0.0010) for r in rows],
            "range_atr": [r.get("range_atr", 0.5) for r in rows],
        },
        index=index,
    )


def synthetic_m5_days(n_days: int, *, seed: int = 7) -> pd.DataFrame:
    """Deterministic multi-day M5 EURUSD-like series (00:00..12:55 NY/day).

    ~156 bars/day at a strict 5-minute cadence, with a natural overnight gap
    between days (so cadence inference must be robust to gaps).
    """
    rng = np.random.default_rng(seed)
    recs: list[dict] = []
    base = pd.Timestamp("2020-01-06 00:00:00", tz=NY_TZ)  # a Monday
    price = 1.1000
    for d in range(n_days):
        day0 = base + pd.Timedelta(days=d)
        for b in range(156):  # 156 * 5min = 13h -> 00:00..12:55
            ts = day0 + pd.Timedelta(minutes=5 * b)
            drift = float(rng.normal(0.0, 0.00035))
            price = max(0.5, price + drift)
            hi = price + abs(float(rng.normal(0.0, 0.00045)))
            lo = price - abs(float(rng.normal(0.0, 0.00045)))
            op = float(rng.uniform(lo, hi))
            recs.append(
                {
                    "timestamp": ts,
                    "open": op,
                    "high": max(hi, op, price),
                    "low": min(lo, op, price),
                    "close": price,
                }
            )
    return make_ohlc_frame(recs)


def make_spy() -> tuple[types.SimpleNamespace, list[tuple]]:
    """A strategy-shaped object that records every (id(frame), len, i)."""
    calls: list[tuple] = []

    def signal(frame, i, params):
        calls.append((id(frame), len(frame), int(i)))
        return None

    spy = types.SimpleNamespace(NAME="spy_contract", WARMUP_BARS=0, signal=signal)
    return spy, calls


class OneShotLong:
    """Signals exactly once at a given NY HH:MM; valid ATR stop (opens a trade)."""

    NAME = "one_shot_long"
    WARMUP_BARS = 0

    @staticmethod
    def signal(frame, i, params):
        if frame.index[i].strftime("%H:%M") == params["signal_hhmm"]:
            return {
                "direction": "long",
                "stop_mode": "atr",
                "stop_atr": 1.0,
                "target_rr": 1.0,
                "session_name": "all_day",
            }
        return None


class LeakyStrategy:
    """Deliberately non-causal: decision depends on a FUTURE bar (i+1)."""

    NAME = "leaky_lookahead"
    WARMUP_BARS = 0

    @staticmethod
    def signal(frame, i, params):
        if i + 1 >= len(frame):
            return None
        # Reads the future close -> must be detected by the harness.
        if float(frame["close"].iloc[i + 1]) > float(frame["close"].iloc[i]):
            return {
                "direction": "long",
                "stop_mode": "atr",
                "stop_atr": 1.0,
                "target_rr": 1.0,
                "session_name": "all_day",
            }
        return None


# ---------------------------------------------------------------------------
# Reusable universal anti-lookahead harness
# ---------------------------------------------------------------------------
def lookahead_leak_indices(strategy_module, frame: pd.DataFrame, params: dict,
                           indices: list[int]) -> list[int]:
    """Return the subset of ``indices`` where the strategy's decision changes
    when every row strictly after ``i`` is poisoned. A causal strategy reads
    only rows ``<= i`` so its output MUST be invariant -> empty list.
    """
    leaks: list[int] = []
    cols = ["open", "high", "low", "close"]
    for i in indices:
        baseline = strategy_module.signal(frame, i, params)
        for poison in (np.nan, 9.9999):
            corrupted = frame.copy()
            corrupted.iloc[i + 1:, [corrupted.columns.get_loc(c) for c in cols]] = poison
            mutated = strategy_module.signal(corrupted, i, params)
            if baseline != mutated:
                leaks.append(i)
                break
    return leaks


class SignalContractTests(unittest.TestCase):
    """TEST GROUP A — signal contract / invocation."""

    def test_engine_contract_exposes_full_frame_and_requires_strategy_causality(self):
        """The engine passes the WHOLE frame + integer i (no causal sandbox).

        Protects against the false belief that the engine slices history for
        the strategy. If this ever changes, the contract docs must change too.
        """
        frame = make_ohlc_frame(
            [
                {"timestamp": f"2022-01-03 10:{m:02d}:00", "open": 1.1, "high": 1.1005,
                 "low": 1.0995, "close": 1.1}
                for m in range(0, 30, 5)
            ]
        )
        spy, calls = make_spy()
        run_backtest(spy, frame, {}, EngineConfig(pair="EURUSD", max_trades_per_day=2),
                     np.zeros(len(frame), dtype=bool), False)

        self.assertTrue(calls, "engine never called strategy.signal()")
        frame_ids = {c[0] for c in calls}
        seen_lengths = {c[1] for c in calls}
        seen_i = sorted(c[2] for c in calls)
        self.assertEqual(len(frame_ids), 1, "engine passed >1 distinct frame object")
        self.assertEqual(seen_lengths, {len(frame)},
                         "engine passed a sliced/partial frame, not the full frame")
        self.assertEqual(seen_i, list(range(spy.WARMUP_BARS, len(frame))),
                         "i must be a contiguous bar index from WARMUP_BARS")
        self.assertTrue(all(0 <= i < len(frame) for i in seen_i),
                        "i must always index into the full frame (never the future end)")

    def test_signal_signature_is_frame_i_params(self):
        """Canonical reference strategy keeps the documented 3-arg signature."""
        params = list(inspect.signature(veorb.signal).parameters)
        self.assertEqual(params[:3], ["frame", "i", "params"],
                         f"signal() contract signature drifted: {params}")

    def test_engine_prefers_generate_signal_when_present(self):
        """engine.py:721-725 fallback contract: generate_signal wins over signal."""
        frame = make_ohlc_frame(
            [
                {"timestamp": f"2022-01-03 10:{m:02d}:00", "open": 1.1, "high": 1.1005,
                 "low": 1.0995, "close": 1.1}
                for m in range(0, 25, 5)
            ]
        )
        gen_calls: list[int] = []
        sig_calls: list[int] = []
        strat = types.SimpleNamespace(
            NAME="dual", WARMUP_BARS=0,
            signal=lambda f, i, p: sig_calls.append(i),
            generate_signal=lambda f, i, p: gen_calls.append(i) or None,
        )
        run_backtest(strat, frame, {}, EngineConfig(pair="EURUSD", max_trades_per_day=2),
                     np.zeros(len(frame), dtype=bool), False)
        self.assertTrue(gen_calls, "generate_signal was not used despite being present")
        self.assertFalse(sig_calls, "signal() must not be called when generate_signal exists")

    def test_entry_fill_is_t_plus_1_never_same_bar(self):
        """Signal at bar i -> fill at bar i+1's OPEN (never the signal bar).

        Zero-cost config so the entry price equals an OHLC open exactly. The
        signal fires on the bar labelled 11:00 (index 1). A T+1 fill must
        execute at the NEXT bar's distinctive open (1.10500), NEVER the signal
        bar's own open (1.10020). Comparing prices avoids the right-labelled
        timestamp ambiguity.
        """
        frame = make_ohlc_frame(
            [
                {"timestamp": "2022-01-03 10:45:00", "open": 1.10000, "high": 1.10040,
                 "low": 1.09980, "close": 1.10000},
                {"timestamp": "2022-01-03 11:00:00", "open": 1.10020, "high": 1.10050,
                 "low": 1.10000, "close": 1.10030},   # signal bar (i=1)
                {"timestamp": "2022-01-03 11:15:00", "open": 1.10500, "high": 1.10560,
                 "low": 1.10490, "close": 1.10520},   # T+1 fill bar (open=1.10500)
                {"timestamp": "2022-01-03 11:30:00", "open": 1.10520, "high": 1.10540,
                 "low": 1.10470, "close": 1.10480},
            ]
        )
        result = run_backtest(
            OneShotLong, frame, {"signal_hhmm": "11:00"},
            EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=0.0,
                         max_spread_pips=10.0, slippage_pips=0.0,
                         commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2),
            np.zeros(len(frame), dtype=bool), False,
        )
        self.assertEqual(len(result.trades), 1)
        entry_price = float(result.trades.iloc[0]["entry_price"])
        # T+1 fill prices inside the NEXT bar's range (~1.1050), tolerating the
        # engine's modelled spread/slippage. A same-bar (lookahead) fill would
        # price at the signal bar's level (<= 1.10050). The ~48-pip gap makes
        # this unambiguous and immune to the cost model.
        self.assertGreater(entry_price, 1.10300,
                           "entry priced at the signal bar => same-bar lookahead")
        self.assertTrue(1.10460 <= entry_price <= 1.10620,
                        f"entry {entry_price} not in the T+1 bar's range (~1.1050)")


class AntiLookaheadHarnessTests(unittest.TestCase):
    """TEST GROUP B — universal anti-lookahead sentinel (non-decorative)."""

    def test_harness_detects_a_known_leaky_strategy(self):
        """The harness MUST catch a strategy that reads frame[i+1]; otherwise
        it would be decorative and give false assurance."""
        frame = synthetic_m5_days(1)
        idx = list(range(10, len(frame) - 2, 17))
        leaks = lookahead_leak_indices(LeakyStrategy, frame, {}, idx)
        self.assertTrue(leaks, "anti-lookahead harness FAILED to detect a real leak")

    def test_ve_orb_is_causal_under_future_poisoning(self):
        """Real strategy VE-ORB must be invariant when rows > i are poisoned
        (it only reads :i, i-1, i, backward rolling, iloc[i-lookback:i])."""
        frame = synthetic_m5_days(4)  # > WARMUP/lookback, multi-day OR windows
        params = dict(veorb.DEFAULT_PARAMS)
        # Indices inside VE-ORB's entry window (08:00..11:59 NY) and >= lookback.
        idx = [
            i for i in range(220, len(frame) - 2)
            if 480 <= (frame.index[i].hour * 60 + frame.index[i].minute) < 720
        ][:40]
        self.assertTrue(idx, "fixture produced no in-window indices to test")
        leaks = lookahead_leak_indices(veorb, frame, params, idx)
        self.assertEqual(leaks, [], f"VE-ORB lookahead detected at indices {leaks}")

    def test_tp01_is_causal_under_future_poisoning(self):
        """Optimized strategy TP-01 must be invariant when rows > i are poisoned."""
        frame = synthetic_m5_days(4)
        params = dict(tp01.DEFAULT_PARAMS)
        idx = [
            i for i in range(250, len(frame) - 2)
            if 420 <= (frame.index[i].hour * 60 + frame.index[i].minute) < 1020
        ][:40]
        self.assertTrue(idx, "fixture produced no in-window indices to test")
        leaks = lookahead_leak_indices(tp01, frame, params, idx)
        self.assertEqual(leaks, [], f"TP-01 lookahead detected at indices {leaks}")


if __name__ == "__main__":
    unittest.main()
