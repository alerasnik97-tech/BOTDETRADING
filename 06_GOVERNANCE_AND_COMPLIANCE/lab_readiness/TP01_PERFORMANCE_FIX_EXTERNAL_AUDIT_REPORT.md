# TP01 PERFORMANCE FIX EXTERNAL AUDIT REPORT

Audit date: 2026-05-16
Auditor mode: Strategy Performance Fix Auditor / no-lookahead specialist / cache-safety auditor / pre-formal-rerun gatekeeper
Audit type: READ-ONLY. No formal backtest, no strategy run, no optimization, no sweep, no validation, no holdout, no 2025/2026, no news, no high precision. No engine/data/strategy/test code modified.

---

## 1. Status

**TP01_PERFORMANCE_FIX_APPROVED_WITH_WARNINGS_READY_FOR_FORMAL_RERUN**

The fix is functionally faithful, causally safe, and converts the O(N²) blocker to O(N). It is cleared for the scoped one-strategy formal train-only 2015–2024 rerun. Approval carries non-blocking warnings (cache-key robustness, unbounded global cache, runtime is minutes not seconds) that must be documented and scheduled for post-rerun hardening — none of them blocks THIS scoped rerun.

---

## 2. Executive Summary

The fix replaces per-call recomputation of `_atr_series()` and `close.iloc[:i].ewm().mean()` (O(N) work inside an O(N) bar loop ⇒ O(N²) overall, the cause of the killed formal run) with a one-time precompute cached in a module-global dict, read in O(1) by index.

Independent findings:

- **Equivalence is provably exact, not merely "tests pass".** ATR is the *same function call* (`_atr_series(frame, atr_period)`) over the full frame as before — only cached and converted to numpy — so ATR values are bit-identical. The EMA optimization is bit-identical because `ewm(adjust=False).mean()` is a purely recursive causal filter: `y[t] = (1-α)·y[t-1] + α·x[t]`, so `y[t]` over the full series equals `y[t]` over the truncated prefix `close[:i]`. The removed `.dropna()` is provably dead code given `_atr_series`'s NaN structure and the pre-existing window guard, and is fail-closed-equivalent (NaN ⇒ `np.percentile` NaN ⇒ `_all_finite` False ⇒ `None`) in any degenerate case. The equivalence test's reference `original_signal` is a faithful replica of the pre-fix logic.
- **No lookahead is introduced by the fix.** Every array access is ≤ `i`; no `i+1`; the EMA reads strictly ≤ `i-1`; the ATR percentile window `[i-lookback:i]` excludes `i`. The future-mutation test passes. The bar-`i` reads (`atr_values[i]`, `close[i]`, `high[i]`, `low[i]`) are **unchanged from the original** — a pre-existing decision-at-close design, not a regression of this fix.
- **Performance is fixed.** Synthetic worst-case benchmark (every bar in-window, hitting `np.percentile`) shows flat per-bar cost (~195–221 µs) across a 40× size increase (5k→200k). O(N²) is definitively eliminated; scaling is linear. 3.6M-bar worst-case projection ≈ 12–13 min; realistic (~1/6 of bars in the 08:00–12:00 window) ≈ a few minutes.
- **Cache key is not robust (WARNING, not a blocker here).** The key `(id(frame), len(frame), frame.index[-1], atr_period, ema_period)` cannot detect in-place mutation of the *same* frame object that preserves length and last timestamp. This staleness was empirically reproduced. It does **not** trigger in the scoped formal rerun because `run_backtest` was verified to never mutate `frame` in-place during the bar loop and the run uses a single frame and single param set.

47 safe unit tests pass, 0 failures. Static safety scan is clean. No prohibited operation was performed.

---

## 3. Diff Surface Audit

Fix commit: `fb529a67c53caff665aeaf0d0cb692aa46abf65d` — "fix: optimize TP01 signal performance without logic changes"

`git show --stat --name-status fb529a67`:

| Status | File | Class |
|---|---|---|
| M | `03_RESEARCH_LAB/research_lab/strategies/tp01_london_ny_momentum_pullback.py` | code (1) |
| A | `03_RESEARCH_LAB/research_lab/tests/test_tp01_performance_equivalence.py` | test (1) |
| A | `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_TP01_PERFORMANCE_FIX_BEFORE_FORMAL_RERUN.md` | docs |
| A | `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_PERFORMANCE_BLOCKER_FIX_REPORT.md` | docs |

- code_files: 1 — `tp01_london_ny_momentum_pullback.py`
- test_files: 1 — `test_tp01_performance_equivalence.py`
- docs: 2
- unexpected_files: **NONE**

No `engine.py`, no `data_loader.py`, no data, no heavy outputs, no ZIP, no root files. Diff surface is **clean and minimal**.

Pre-existing dirty working tree (NOT introduced by, related to, or staged by this audit): modified/untracked markdown + CSV under `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/{claude_audit,hypothesis_backlog,index}/`. These are documents only — no code/engine/data/TP01 — and are explicitly excluded from all staging in this audit.

---

## 4. TP-01 Code Audit

File: `03_RESEARCH_LAB/research_lab/strategies/tp01_london_ny_momentum_pullback.py`

- `DEFAULT_PARAMS` (lines 14–27): **unchanged** by the diff (no `+`/`-` on those lines). No new or ghost params. The fix only adds a local `ema_period = int(p["ema_period"])` read from the existing param inside `signal()`.
- Output contract (`_build_signal`, lines 77–103): **unchanged**. Same dict keys/values.
- No file I/O, no network, no external dependency, no news, no high-precision, no 2025/2026, no holdout, no new engine dependency. Pure numpy/pandas in-memory.
- `_atr_series` (61–74): **unchanged**. TR via `concat([h-l, |h-prevc|, |l-prevc|]).max(axis=1)` (skipna ⇒ TR[0] valid), then `rolling(period, min_periods=period).mean()` ⇒ ATR is NaN for indices `0 .. atr_period-2`, valid from `atr_period-1`.
- `_get_cached_indicators` (109–125): computes the *same* `_atr_series(frame, atr_period)` plus `frame["close"].astype(float).ewm(span=ema_period, adjust=False).mean()`, stores both as `float64` numpy arrays in module-global `_CACHE`.
- `signal` (128–192): added guards `if i < lookback + atr_period + momentum_bars` (pre-existing) and `if i < ema_period + 2` (replaces the original `if len(ema) < ema_period + 2`, which is exactly equivalent since `len(close[:i]) == i`). Reads `current_atr = atr_values[i]`, `previous_atr_window = atr_values[i-lookback:i]`, `ema_now = ema_values[i-1]`, `ema_prev = ema_values[i-2]`. Removed the original `.dropna()` + `len < lookback` guard.

DEFAULT_PARAMS change: **NO**. Output contract change: **NO**. New external dependency: **NO**.

---

## 5. Cache Safety Audit

Cache key: `(id(frame), len(frame), frame.index[-1] if len>0 else None, atr_period, ema_period)`.

| Property | Verdict |
|---|---|
| cache_key_safe | **WARNING** |
| stale_cache_risk | **MEDIUM** in general / **LOW** for the scoped single formal rerun |
| memory_growth_risk | **LOW** for the scoped rerun / **MEDIUM** as a latent code-quality issue |
| engine_mutates_frame | **NO** (verified) |

Findings:

1. **In-place same-object mutation is invisible to the key (empirically confirmed, Block 6).** A frame mutated in-place on past bars while keeping `id`, `len`, and `index[-1]` constant produced a stale cache hit: `ema[399]` stayed at the stale `1.08949655` while a fresh recompute on the mutated frame gave `1.08951474`. The EMA (infinite-memory recursive filter) is the most exposed; the windowed ATR less so. Hazard reproduced ⇒ Alert #1 is real.
2. **`id(frame)` is not a stable identity across runs.** CPython reuses memory addresses after GC; a different later frame can receive the same `id()`. With an unbounded module-global `_CACHE` that is never cleared, a cross-run collision is theoretically possible if `(len, index[-1], atr_period, ema_period)` also coincide and interior values differ.
3. **Decisive mitigation for the scoped rerun.** `run_backtest` (engine.py:578–707) extracts `open/high/low/close/atr14/range_atr` via `.to_numpy()` once *before* the loop and then iterates `for i in range(WARMUP_BARS, len(frame))` calling `signal(frame, i, params)` on a single, stable, **read-only** frame object. No `frame[...] =`, no `.loc/.iloc[...] =`, no `inplace=` write to `frame` exists anywhere in `run_backtest`. Therefore, within one strategy run the key is stable and correct, the cache is populated once and correctly reused, and no staleness occurs. The formal rerun uses one frame and one param set ⇒ exactly one cache entry.
4. **Memory.** Two `float64` arrays of length N; benchmark shows 3.2 MB at N=200k ⇒ ≈ 57.6 MB at 3.6M for the single rerun entry — acceptable. No eviction/bound/`clear()` helper/documentation ⇒ latent unbounded growth across many runs in a long-lived process (out of scope for this rerun).

Net: acceptable **WITH WARNINGS** for the scoped formal train-only rerun; not a blocker for this task. Hardening recommended post-rerun (see §11).

---

## 6. Equivalence Audit

Reference fidelity (`test_tp01_performance_equivalence.py::original_signal`): **faithful**. It imports and calls the *real* `_atr_series` and `DEFAULT_PARAMS`; the session window, `_all_finite`, and `_build_signal` are inlined with logic mathematically identical to the production helpers; the ATR `.dropna()`+`len` guard, the `close.iloc[:i].ewm(adjust=False)` EMA, the `len(ema) < ema_period+2` guard, momentum/tolerance/buffer math, long/short bias, and entry conditions all match the pre-fix code reconstructed from the diff.

Independent equivalence proof:

- **ATR:** identical function call over the full frame in both old and new code ⇒ bit-identical (only cached + `to_numpy`).
- **EMA:** `ewm(adjust=False).mean()` is `y[0]=x[0]`, `y[t]=(1-α)y[t-1]+α x[t]` — strictly causal. `y[t]` over `close[0:N]` equals `y[t]` over `close[0:i]` (i>t) with the identical float-op sequence ⇒ `ema_values[i-1] == ema.iat[-1]`, `ema_values[i-2] == ema.iat[-2]`, bit-identical, no lookahead.
- **`.dropna()` removal:** guard `i ≥ lookback+atr_period+momentum_bars` ⇒ window start `i-lookback ≥ atr_period+momentum_bars ≥ atr_period-1` (first valid ATR index) for any `momentum_bars ≥ 0` ⇒ window is always NaN-free ⇒ dropna was dead code. Degenerate NaN case is fail-closed-equivalent (both paths return `None`).

Test execution: `python -m unittest ... test_tp01_performance_equivalence.py -v` ⇒ **5/5 OK** (1.585 s). `test_tp01_signals_equivalent_before_after_on_synthetic_cases` asserts full-dict equality for i=250..599 and passes ⇒ empirically confirms the bit-identical result.

Classification: **EQUIVALENCE_STRONG**.

---

## 7. No-Lookahead Audit

| Access | Indices used | Verdict |
|---|---|---|
| `current_atr = atr_values[i]` | TR/ATR over ≤ i (incl. close[i]) | Unchanged from original; decision-at-close — see note |
| `previous_atr_window = atr_values[i-lookback:i]` | i-lookback .. i-1 (excludes i) | No lookahead |
| `ema_now = ema_values[i-1]` | ≤ i-1 (causal EWM) | No lookahead |
| `ema_prev = ema_values[i-2]` | ≤ i-2 | No lookahead |
| `close/high/low[i]`, `prev_*[i-1]`, `close[i-momentum_bars-1]` | ≤ i | Unchanged from original |
| any `i+1` / forward | none | None present |

- future_mutation_test: `test_tp01_no_lookahead_shifted_features` mutates bars `[i+1:]` (with cache cleared) and asserts identical signal ⇒ **PASS**.
- The bar-`i` reads (`atr_values[i]`, `close[i]`, `high[i]`, `low[i]`) are byte-for-byte the same as the pre-fix original. They constitute a *pre-existing* decision-computed-at-close-of-bar-i pattern. They are legitimate **iff** the engine evaluates the closed bar `i` and enters on the next bar. This is **not introduced or changed by this fix** and is therefore out of scope as a fix regression; flagged for completeness (W4).

Classification: **PASS_NO_LOOKAHEAD** for the fix delta (no regression). Pre-existing bar-i semantics: WARNING-level, requires engine next-bar-entry confirmation, not a blocker for this fix audit.

---

## 8. Performance Benchmark Audit

Worst-case synthetic benchmark (Python 3.14.3): all bars forced in-window so every iteration hits the `np.percentile` path. Cache cleared per size.

| N | bars iterated | total | per-bar | cache arrays | proj. 3.6M |
|---|---|---|---|---|---|
| 5,000 | 4,700 | 1.038 s | 220.8 µs | 0.080 MB | ~795 s |
| 50,000 | 49,700 | 10.895 s | 219.2 µs | 0.800 MB | ~789 s |
| 200,000 | 199,700 | 38.945 s | 195.0 µs | 3.200 MB | ~702 s |

Re-run smoke (from the suite): 4,700 bars in 0.8288 s (matches reported 0.8117 s).

Interpretation:

- **Per-bar cost is flat (~195–221 µs) across a 40× size increase ⇒ O(N), the O(N²) is definitively eliminated.** Under O(N²), per-bar would scale ~linearly with N (50k ≈ 10× slower/bar than 5k); it does not. Decisive proof the fix works.
- Worst-case 3.6M projection ≈ **12–13 min** (every bar in 08:00–12:00). Real M1/M5 data has ~4h/24h ≈ 1/6 of bars in-window hitting `np.percentile`; the rest return O(1) at `_in_window`. Realistic signal-generation time ≈ **a few minutes**, plus a one-time O(N) indicator precompute (a few seconds). Memory ≈ 57.6 MB (one entry).
- Residual: `np.percentile` over `lookback=200` runs on every in-window bar ⇒ O(N·lookback) with constant lookback ≈ O(N) at a ~200× constant. Not "seconds", but bounded, linear, and a massive improvement over the previously non-terminating O(N²).

Classification: **PERFORMANCE_PASS_WITH_RUNTIME_WARNING**.

---

## 9. Static Safety Scan

`Grep` over both changed files for `read_csv|to_csv|read_parquet|to_parquet|open(|Path(|os.path|subprocess|requests|http|forex_factory|news|high_precision|level2|holdout|sealed_holdout|2025|2026|rv5|rv15|p30|0.08|zip|000_PARA_CHATGPT`:

- `tp01_london_ny_momentum_pullback.py`: **No matches**
- `test_tp01_performance_equivalence.py`: **No matches** (synthetic timestamps use 2023-01-01)

| Check | Result |
|---|---|
| file_io | NONE |
| news | NONE |
| high_precision | NONE |
| holdout / sealed_holdout | NONE |
| 2025_2026 | NONE |
| ghost_params (rv5/rv15/p30/0.08) | NONE |
| network/subprocess | NONE |

Static scan **clean**. No prohibited dependency introduced.

---

## 10. Tests

Environment: `PYTHONPATH=03_RESEARCH_LAB`, Python 3.14.3. No formal backtest, no strategy run.

| Suite | Result |
|---|---|
| `test_tp01_performance_equivalence.py` | **5/5 OK** (equivalence, no-lookahead, smoke, cache-reuse, cache-partition) |
| `test_priority_a_skeletons.py` | **16/16 OK** |
| `test_engine.py` | **17/17 OK** |
| `test_lab_preflight_no_leakage.py` | **6/6 OK** |
| `test_engine_stop_entry.py` | **3/3 OK** |
| `import research_lab` | OK |
| `STRATEGY_REGISTRY` | 67 strategies; `tp01_london_ny_momentum_pullback` present |
| `import research_lab.engine` | OK |

Total: **47 unit tests, 0 failures.**

---

## 11. Warnings

- **W1 — Cache key cannot detect in-place same-object mutation (MEDIUM general / LOW scoped).** Empirically reproduced staleness. Mitigated for the scoped rerun because the engine is read-only on `frame` and only one frame/param is used. Recommended post-rerun hardening: bounded LRU + explicit cache-clear at run boundary, or key on a content hash / monotonic frame version instead of `id()`.
- **W2 — Unbounded module-global `_CACHE` (MEDIUM latent).** No eviction/bound/`clear_cache()` helper/docstring. ~57.6 MB for the single 3.6M run is fine; unbounded growth only across many runs in a long-lived process. Recommended post-rerun: bounded cache + public `clear_cache()` + docstring documenting key semantics and the mutation contract.
- **W3 — Runtime is minutes, not seconds (LOW).** `np.percentile` over the 200-bar lookback runs every in-window bar. Acceptable and linear; the operator must record the real wall-clock runtime of the formal rerun.
- **W4 — Pre-existing bar-`i` reads (informational, not a fix regression).** `current_atr/close/high/low` at `i` require the engine to evaluate the closed bar `i` and enter next-bar. Unchanged from the original; confirm engine next-bar-entry semantics independently of this fix.
- **W5 — Equivalence-suite coverage gap (LOW).** No same-object in-place-mutation staleness test and no NaN-injected ATR-window divergence test. Outcomes are provably equivalent / both-`None` anyway; a hardening regression test is recommended alongside W1/W2.
- **W6 — Fix report wording.** `TP01_PERFORMANCE_BLOCKER_FIX_REPORT.md` calls `_CACHE` "thread-safe"; a plain dict with no lock is not thread-safe. Irrelevant for the single-threaded backtest but the claim should be corrected.

None of W1–W6 blocks the scoped one-strategy formal train-only 2015–2024 rerun.

---

## 12. Decision

**TP01_PERFORMANCE_FIX_APPROVED_WITH_WARNINGS_READY_FOR_FORMAL_RERUN.**

Rationale: equivalence is provably exact and empirically confirmed; no lookahead is introduced; the O(N²) blocker is converted to O(N) (linear scaling proven); diff surface, static scan, and 47 safe tests are clean; the only material risk (cache staleness, W1) is structurally prevented in the scoped rerun by the engine's verified read-only treatment of `frame` and the single-frame/single-param usage. Proceed to the scoped formal train-only rerun; schedule W1/W2/W5 hardening for after the rerun (no code changes in this phase).

---

## 13. Safety Verification

- formal_backtest_run: NO
- strategy_run_real: NO
- optimization_run: NO
- sweep_run: NO
- validation_run: NO
- holdout_used: NO
- 2025_2026_used: NO
- news_used: NO
- high_precision_used: NO
- engine_modified: NO
- data_modified: NO
- force_push: NO
- git_add_dot_used: NO

---

## 14. Copy-Paste Summary for ChatGPT

```
TP01 PERFORMANCE FIX — EXTERNAL AUDIT RESULT

STATUS: TP01_PERFORMANCE_FIX_APPROVED_WITH_WARNINGS_READY_FOR_FORMAL_RERUN

Fix commit fb529a67 audited READ-ONLY (no formal run, no strategy run, no opt/sweep/
validation/holdout, no 2025-2026, no news, no high precision; engine/data/strategy/tests
NOT modified).

EQUIVALENCE: STRONG. ATR = same function call, bit-identical (just cached). EMA bit-
identical because ewm(adjust=False) is a causal recursion: value at t over full series
== value at t over close[:i]. Removed .dropna() is provably dead code under the existing
guard and fail-closed-equivalent on NaN. Reference original_signal is a faithful replica.
Equivalence test asserts full-dict equality i=250..599 -> PASS.

NO-LOOKAHEAD: PASS for the fix delta. All accesses <= i, no i+1, EMA <= i-1, ATR window
[i-lookback:i] excludes i. future-mutation test passes. Bar-i reads are unchanged from
the original (pre-existing decision-at-close; needs engine next-bar-entry semantics, not
a regression of this fix).

PERFORMANCE: O(N^2) ELIMINATED. Per-bar cost flat ~195-221us across 5k/50k/200k (40x
size, linear). 3.6M worst-case ~12-13 min; realistic (~1/6 bars in-window) ~ a few min.
Memory ~57.6 MB single run.

CACHE: WARNING (not a blocker here). Key (id(frame),len,index[-1],atr_p,ema_p) cannot
detect in-place same-object mutation -> staleness reproduced empirically. NOT triggered
in the scoped rerun: engine run_backtest is read-only on frame (verified) and uses one
frame/one param. Harden post-rerun (bounded cache + clear helper + content/version key).

TESTS: 47 safe unit tests pass, 0 failures (tp01-equivalence 5, priority_a 16, engine 17,
preflight 6, stop_entry 3) + import/registry OK (67 strategies, tp01 present).

STATIC SCAN: clean (no file IO/news/high_precision/holdout/2025-2026/ghost params).

DECISION: APPROVED WITH WARNINGS -> proceed to scoped TP01 formal train-only 2015-2024
rerun (one strategy, no opt/sweep/holdout/2025-2026/news/high-precision; record real
runtime). Schedule W1/W2/W5 cache hardening AFTER the rerun.
```
