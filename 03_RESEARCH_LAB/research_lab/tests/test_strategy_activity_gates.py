"""Zero-activity sentinel + performance-complexity guard.

Addresses two systemic gaps from
``ENGINE_STRATEGY_CONTRACT_AUDIT_VEORB_V1.md``:

  * C9 — the reconciliation gate checks metric *consistency*, NOT signal
    *density*. A strategy that emits ~0 trades (cadence-inference None, an
    over-restrictive filter, or a bug) produces a clean, gate-passing dossier
    indistinguishable from "regime obsolete" (exactly VE-ORB's shape:
    15 trades, all in 2015-01..2015-02 over a 2015-2024 window).
  * C7 — VE-ORB's ``signal()`` is O(N^2) (per-call full-frame ATR recompute +
    per-call O(i) opening-range scan), uncaught by any gate.

``assess_activity`` is a pure, dependency-free helper kept test-local on
purpose: this phase does NOT modify engine/runner core. Wiring it into the
production seal gate is deferred (reported as
WARN_ZERO_ACTIVITY_SENTINEL_NOT_WIRED).
"""
from __future__ import annotations

import re
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab.config import NY_TZ
from research_lab.strategies import ve_orb_volatility_expansion as veorb


# ---------------------------------------------------------------------------
# Zero-activity sentinel (pure helper)
# ---------------------------------------------------------------------------
def assess_activity(trade_timestamps, period_start: str, period_end: str,
                    *, min_coverage_ratio: float = 0.05,
                    max_single_month_share: float = 0.90) -> dict:
    """Classify a strategy's activity over a declared period.

    A 'degenerate' strategy must NOT be silently archived as 'regime obsolete'
    without an explicit human note.
    """
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    period_months = max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)
    period_years = max(1, end.year - start.year + 1)

    ts = pd.to_datetime(pd.Series(list(trade_timestamps)), errors="coerce").dropna()
    total = int(ts.shape[0])

    if total == 0:
        return {
            "total_trades": 0, "distinct_years": 0, "distinct_months": 0,
            "coverage_ratio": 0.0, "max_single_month_share": 0.0,
            "zero_activity": True, "single_year_only": False,
            "extreme_concentration": False, "is_degenerate": True,
            "flags": ["zero_activity"],
        }

    years = ts.dt.year
    ym = ts.dt.year * 12 + ts.dt.month
    distinct_years = int(years.nunique())
    distinct_months = int(ym.nunique())
    coverage_ratio = distinct_months / period_months
    top_month_share = float(ym.value_counts(normalize=True).iloc[0])

    single_year_only = distinct_years == 1 and period_years > 1
    low_coverage = coverage_ratio < min_coverage_ratio
    extreme_concentration = top_month_share >= max_single_month_share

    flags: list[str] = []
    if single_year_only:
        flags.append("single_year_only")
    if low_coverage:
        flags.append("low_coverage")
    if extreme_concentration:
        flags.append("extreme_concentration")

    return {
        "total_trades": total, "distinct_years": distinct_years,
        "distinct_months": distinct_months,
        "coverage_ratio": round(coverage_ratio, 4),
        "max_single_month_share": round(top_month_share, 4),
        "zero_activity": False, "single_year_only": single_year_only,
        "extreme_concentration": extreme_concentration,
        "is_degenerate": bool(flags),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Performance-complexity static scanner (heuristic lint)
# ---------------------------------------------------------------------------
_QUADRATIC_PATTERNS = {
    "prefix_slice_iloc[:i]": re.compile(r"\.iloc\[\s*:\s*i\s*\]"),
    "per_call_index_iteration": re.compile(r"for\s+\w+\s+in\s+[\w.]*\.index\b"),
    "full_frame_atr_recompute": re.compile(r"_atr_series\(\s*frame\b"),
}


def scan_quadratic_risk(source: str) -> list[str]:
    """Return names of O(N^2)-suggestive patterns found in strategy source."""
    return sorted(name for name, pat in _QUADRATIC_PATTERNS.items() if pat.search(source))


def _synthetic_m5(n_bars: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-06 00:00:00", tz=NY_TZ)
    idx, op, hi, lo, cl = [], [], [], [], []
    price = 1.10
    for b in range(n_bars):
        # 156 bars/day then jump to next day (overnight gap, strict 5-min cadence).
        day, within = divmod(b, 156)
        ts = base + pd.Timedelta(days=day) + pd.Timedelta(minutes=5 * within)
        price = max(0.5, price + float(rng.normal(0, 0.0003)))
        h = price + abs(float(rng.normal(0, 0.0004)))
        low = price - abs(float(rng.normal(0, 0.0004)))
        idx.append(ts); op.append(price); hi.append(h); lo.append(low); cl.append(price)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": cl},
                        index=pd.DatetimeIndex(idx))


class ZeroActivitySentinelTests(unittest.TestCase):
    """TEST GROUP C."""

    def test_zero_trades_is_flagged_degenerate(self):
        r = assess_activity([], "2015-01-01", "2024-12-31")
        self.assertTrue(r["zero_activity"])
        self.assertTrue(r["is_degenerate"])

    def test_veorb_shape_15_trades_all_in_2015_is_flagged(self):
        """The exact VE-ORB incident shape must trip the sentinel."""
        jan = [f"2015-01-{d:02d} 09:00:00" for d in (6, 8, 9, 12, 16, 19, 20, 21, 22, 27, 29)]
        more_jan = ["2015-01-09 10:15:00", "2015-01-16 09:10:00", "2015-01-20 11:00:00"]
        feb = ["2015-02-02 11:35:00"]
        r = assess_activity(jan + more_jan + feb, "2015-01-01", "2024-12-31")
        self.assertEqual(r["total_trades"], 15)
        self.assertTrue(r["single_year_only"])
        self.assertTrue(r["extreme_concentration"])
        self.assertTrue(r["is_degenerate"],
                        "VE-ORB-shaped activity must NOT pass as healthy")

    def test_healthy_distribution_is_not_flagged(self):
        ts = []
        for year in range(2015, 2025):
            for month in range(1, 13):
                ts.append(f"{year}-{month:02d}-15 10:00:00")
                ts.append(f"{year}-{month:02d}-20 11:00:00")
        r = assess_activity(ts, "2015-01-01", "2024-12-31")
        self.assertFalse(r["is_degenerate"], f"healthy series misflagged: {r['flags']}")
        self.assertEqual(r["distinct_years"], 10)


class PerformanceComplexityGuardTests(unittest.TestCase):
    """TEST GROUP D."""

    def test_scanner_flags_veorb_known_quadratic_pattern(self):
        """Characterization guard: VE-ORB's documented O(N^2) shape is present.

        If a future refactor removes the per-call prefix scan / full-frame ATR
        recompute, this test fails on purpose -> update the contract docs.
        """
        src = Path(veorb.__file__).read_text(encoding="utf-8")
        hits = scan_quadratic_risk(src)
        self.assertIn("prefix_slice_iloc[:i]", hits)
        self.assertIn("full_frame_atr_recompute", hits)

    def test_scanner_is_quiet_on_a_clean_o1_strategy(self):
        """The lint must not fire on a per-bar O(1) strategy (non-decorative)."""
        clean = (
            "def signal(frame, i, params):\n"
            "    if i < 2: return None\n"
            "    if frame['close'].iat[i] > frame['close'].iat[i-1]:\n"
            "        return {'direction': 'long'}\n"
            "    return None\n"
        )
        self.assertEqual(scan_quadratic_risk(clean), [])

    def test_veorb_signal_smoke_completes_within_generous_budget(self):
        """Catastrophic-regression guard only (stable, generous bound).

        Not a benchmark: it just ensures a single signal() call on a modest
        frame returns quickly enough that batches remain feasible.
        """
        frame = _synthetic_m5(900)
        params = dict(veorb.DEFAULT_PARAMS)
        i = next(
            k for k in range(220, len(frame))
            if 480 <= frame.index[k].hour * 60 + frame.index[k].minute < 720
        )
        t0 = time.perf_counter()
        out = veorb.signal(frame, i, params)
        elapsed = time.perf_counter() - t0
        self.assertTrue(out is None or isinstance(out, dict),
                        f"signal() returned an invalid type: {type(out)}")
        self.assertLess(elapsed, 10.0, f"signal() took {elapsed:.2f}s on 900 bars")


if __name__ == "__main__":
    unittest.main()
