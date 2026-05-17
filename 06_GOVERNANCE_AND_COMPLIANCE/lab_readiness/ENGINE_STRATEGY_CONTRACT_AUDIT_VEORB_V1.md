# ENGINE STRATEGY CONTRACT AUDIT — VEORB V1

> Read-only static audit of the runner ↔ engine ↔ data-feed ↔ `signal()` ↔
> strategy contract. No code/data/runner/engine/strategy modified. No backtest,
> validation, holdout, 2025/2026, optimization or sweep executed. No ZIP. No
> `git add .`. This audit does NOT revive VE-ORB and does NOT approve any
> strategy for validation/holdout/production.

---

## 1. Audit Status

**AUDIT_PASS_WITH_PERFORMANCE_WARNINGS_VEORB_REJECT**

The engine/runner execution path is **causally sound** (T+1 fills, no
future-leak in execution, full-period iteration, fail-closed scope safety) and
the VE-ORB run was **faithfully executed**. No validity/lookahead bug exists in
this run. However, the engine↔strategy *contract* delegates 100% of causality
and performance discipline to each strategy with **no engine-level guardrail or
universal test**, and VE-ORB's logic is O(N²). These are **MEDIUM systemic
risks** for the 125+ strategy population and future batches — not a blocker for
this run. VE-ORB remains rejected/non-viable. A lightweight contract-test phase
is warranted before large new batches (future prompt created).

## 2. Executive Verdict

The decisive question — *does the engine feed `signal()` a sliced history or the
whole multi-year frame?* — is answered: **engine.py:725 passes the entire
`frame` plus the integer index `i`**. The engine performs **no history slicing
and no causal enforcement**; every strategy is trusted to read only rows `≤ i`.

For VE-ORB this is **safe**: it only reads `frame.iloc[:i]`, `iat[i-1]`,
`iat[i]`, a backward-only rolling ATR, and `iloc[i-lookback:i]` — no future
access. The frame index is **NY-localized by the data loader**
(`parse_prepared_index → tz_convert(NY_TZ)`), so VE-ORB's 07:00/08:00/12:00
windows are NY-local and DST-safe and match the reported `entry_time_ny`. The
engine loop `for i in range(WARMUP_BARS, len(frame))` (engine.py:713) iterates
**every** bar 2015→2024 with **no timeout/early-stop**; equity (728,997 rows ≈
prepared M5 729,382) confirms the full series was processed. **This definitively
closes every residual technical doubt from the prior diagnostic**: there is no
engine cutoff, no timezone bug, no cadence flip, no data truncation — VE-ORB's
zero trades after 2015-02-02 are genuinely its own `signal()` returning `None`.

The systemic concern is the *contract itself*: passing the full frame with zero
guardrail makes silent lookahead trivial to introduce in a less disciplined
strategy; the O(N²) `_opening_range`/ATR-recompute pattern can make formal
batches infeasibly slow; and a strategy that silently emits zero signals
produces a clean, gate-passing dossier indistinguishable from "regime obsolete".
None of these failed *here*, but the lab has no universal test that would catch
them in the next strategy.

## 3. Scope

| Item | Value |
|---|---|
| base branch | `diagnostic/veorb-zero-trades-2016-2024-20260517` @ `3061e4dcdb232ca4843713680e1f0091415f5083` |
| audit branch | `audit/engine-strategy-contract-veorb-v1-20260517` |
| files inspected (read-only) | `runners/formal_train_runner.py` (full), `engine.py` (≈595–765 + signal-call grep), `data_loader.py` (index/timeframe grep), `strategies/ve_orb_volatility_expansion.py` (full, prior turn), `prepared_data_manifest.json` (prior turn), test inventory grep (`tests/`) |
| no rerun | **Confirmed.** Only `git`, `Read`, `Glob`, `Grep`. No backtest, no `--execute`, no validation/holdout/2025-2026/optimization/sweep. |

## 4. Safety Verification

| Control | Result |
|---|---|
| code modified? | **NO** |
| data modified? | **NO** |
| backtest run? | **NO** |
| validation run? | **NO** |
| holdout used? | **NO** |
| 2025/2026 used? | **NO** |
| optimization / sweep? | **NO** |
| heavy outputs committed? | **NO** (only the two governance docs staged) |
| `git add .` used? | **NO** (explicit per-file add) |
| force push / merge / rebase / reset --hard / clean / stash? | **NO** |
| preexisting dirty tree touched? | **NO** (unrelated `external_research_20260516/` left untouched) |

## 5. Signal Contract Audit

| component | observed_contract | evidence_file | risk | severity |
|---|---|---|---|---|
| invocation | `strategy_module.signal(frame, i, params)` (or `generate_signal`) | engine.py:721–725 | full frame + int index; no slicing | WARN (systemic) |
| frame passed | the **entire** multi-year DataFrame (not `iloc[:i]`) | engine.py:599,725 | strategy must self-restrict to `≤ i` | WARN (systemic lookahead surface) |
| causal enforcement | **none** by engine | engine.py:713–731 | engine trusts strategy discipline | WARN (systemic) |
| execution model | signal@`i`, fill@`i+1` (T+1) | engine.py:739 | execution side is causal | PASS |
| iteration | `for i in range(WARMUP_BARS, len(frame))`, no timeout/early-stop | engine.py:713 | full period processed | PASS |
| gating | signal only if flat, no pending, past cooldown, < max_trades/day | engine.py:719–720 | correct; VE-ORB silence is its own `None` | PASS |
| interface | module exposes `NAME`, `WARMUP_BARS`, `DEFAULT_PARAMS`, `signal(frame,i,params)`; optional `generate_signal` | runner.py:449–486; engine.py:611,721–725 | duck-typed, **undocumented**, no formal Protocol | WARN |

Classification: **`WARN_SIGNAL_CONTRACT_UNDOCUMENTED` +
`WARN_SIGNAL_CONTRACT_PERFORMANCE_RISK`**. **Not**
`BLOCKER_SIGNAL_CONTRACT_LOOKAHEAD_RISK` (no lookahead in this run; execution is
T+1 causal) and **not** `BLOCKER_SIGNAL_CONTRACT_INCONSISTENT` (invocation is
deterministic and consistent).

## 6. Index / Timezone / Cadence Audit

- **Index/timezone:** data loader sets the frame index via
  `parse_prepared_index` → `tz_convert(NY_TZ)` (data_loader.py:58–62, applied
  132 & 152). The engine independently re-derives a NY index for its own
  session/exit logic (engine.py:636, DST-aware via `tz_convert`). The strategy
  receives the **NY-localized** `frame.index`; VE-ORB `_minute_of(frame.index[i])`
  is therefore NY-local wall-clock → DST-safe and consistent with reported
  `entry_time_ny` (08:00–11:35). → **PASS_TIME_CONTRACT_SAFE** for VE-ORB.
- **DST:** handled by `tz_convert` on both loader and engine sides; local
  wall-clock windows are stable across DST. No shift bug. (Thin point: relies on
  the loader; now documented here.)
- **Cadence:** VE-ORB `_infer_cadence_minutes` = median of **all prefix
  deltas**; for a ~5-min series with weekend/holiday gaps the median is robustly
  5.0 across the whole 2015–2024 frame → cadence stable, never flips per-year.
  Not a cause of the cliff. **But** it returns `None` (→ all signals suppressed,
  silently) if the median rounds outside `{1,2,3,5,10,15}` or the index has
  duplicates — a fragile, silent-kill design pattern. → **`WARN_CADENCE_INFERENCE_FRAGILE`**.
- **Timeframe traceability defect:** runner calls
  `load_backtest_data_bundle(..., target_timeframe="M1")`
  (formal_train_runner.py:454–457) yet the iterated frame is ~M5-scale
  (equity 728,997 ≈ prepared M5 729,382, not M1 3.63M), trades stamp
  `data_source_used=prepared_m5_bid` (an engine **default label** when `None`,
  engine.py:650 — not proof of timeframe), and the summary says M15. The
  effective strategy timeframe is **not unambiguously pinned by the artifacts**.
  → `WARN_TIMEFRAME_TRACEABILITY` (no correctness impact on the zero-trades
  finding; data spans 2015–2024 regardless).

## 7. Opening Range Logic Audit

`_opening_range` (strategy lines 120–138) + `_or_window_is_complete`
(87–117): input `rows = frame.iloc[:i]` (strictly past, **excludes** current
bar — no lookahead); filters to `idx.date()==ts.date()` AND minute ∈
[07:00,08:00); requires ≥`ceil(12·0.9)=11` unique bars, first ≤07:05, last
≥07:55; rejects NaN. **Correctness: safe** (no future scan, no date-filter bug,
recomputed per bar deterministically). **Performance: pathological** — a Python
list comprehension over **all** of `rows.index` every call, and `_atr_series`
recomputes ATR over the **entire** frame every call → **O(N²)** over ~729k bars.
→ **`WARN_OPENING_RANGE_PERFORMANCE_RISK`** (not
`BLOCKER_OPENING_RANGE_LOOKAHEAD`, not `BLOCKER_OPENING_RANGE_DATE_FILTER_BUG`).

## 8. Cadence Inference Audit

`_infer_cadence_minutes` (strategy lines 73–84): median of all positive prefix
deltas; `None` if `<3` rows / duplicates / rounded ∉ `{1,2,3,5,10,15}` /
`|cadence−round|>0.01`. Recomputed over the **growing multi-year prefix** every
call (O(N) per call → contributes to O(N²)). For this run cadence is a stable
`5`. Risk is **fragility + silent total suppression**, not a per-year bug. →
**`WARN_CADENCE_INFERENCE_FRAGILE`** (not `BLOCKER_CADENCE_INFERENCE_BUG`).

## 9. Lookahead Risk Review

- **Engine execution:** causal — signal at `i`, fill at `i+1` (engine.py:739);
  exits use per-bar arrays indexed at/<= current bar. No future leak in the
  harness.
- **Contract surface:** the engine hands the **whole frame** to `signal()` with
  no causal sandbox → lookahead is *trivially introducible* by any strategy
  (`frame['x'].mean()`, `.shift(-1)`, `frame.iloc[i+1:]`, full-frame fit). The
  engine would **not** detect it.
- **VE-ORB specifically:** audited line-by-line — **no lookahead** (only `:i`,
  `i-1`, `i`; backward rolling; past percentile window).
- **Test coverage:** a no-lookahead test exists but is **TP01-specific**
  (`tests/test_tp01_performance_equivalence.py:128–147`, "mutate future bars →
  signal must not change"). There is **no universal, strategy-agnostic
  no-lookahead contract test** and none covering VE-ORB. → systemic gap.

## 10. Performance Complexity Review

VE-ORB `signal()` is **O(N²)** for the full series (per-call full-frame ATR
recompute + per-call O(i) OR list-comprehension). At ~729k bars this is
~10¹¹–10¹² operations. The engine imposes **no per-bar timeout / no
max-iterations**, so correctness is preserved (the run completed; equity spans
all years) at the cost of extreme runtime. This pattern is **easy to write,
uncaught by any gate**, and a single such strategy can make a 3-profile formal
batch infeasibly slow. Systemic performance hazard for future batches.

## 11. Systemic Risk For Future Strategies

1. VE-ORB-specific? The O(N²) coding is VE-ORB's; the **enabling contract**
   (full frame, no guardrail) is **engine-wide** → affects all 125+ strategies.
2. Engine/runner contract can affect other strategies? **Yes** — same
   `signal(frame,i,params)` path for every strategy.
3. Systemic lookahead risk? **Yes, latent** — no engine sandbox, no universal
   test; only ad-hoc per-strategy checks.
4. Systemic performance risk? **Yes** — no timeout; O(N²) patterns uncaught.
5. Systemic timezone risk? **Low** — loader+engine both `tz_convert(NY)`,
   DST-safe; documented thin-point only.
6. Silent zero-trade risk? **Yes** — a strategy emitting 0 signals (cadence
   `None`, over-restrictive filter, or a bug) yields a clean, **gate-passing**
   dossier indistinguishable from "regime obsolete" (exactly VE-ORB's shape).
   The reconciliation gate checks metric *consistency*, **not signal density**.
7. Tests sufficient? **No** — runner fail-closed/scope tests are good; there is
   no universal no-lookahead test and no zero-activity sentinel.
8. Contract tests needed before new batches? **Yes** (lightweight, no backtest).

**Systemic risk level: `SYSTEMIC_RISK_MEDIUM`** — latent, not a blocker (the
engine execution is causal and this run was faithful), but a real hazard for the
strategy population and future batches.

## 12. Test Coverage Gaps

| gap | current state | needed (lightweight, no backtest) |
|---|---|---|
| Universal no-lookahead | only TP01-specific (test_tp01:128) | generic "future-bar mutation must not change `signal()@i`" applied to every registered strategy on a tiny synthetic frame |
| Zero-activity sentinel | none (gate checks metric consistency only) | assertion/telemetry that flags a profile with ~0 signals over the period before "regime obsolete" labeling |
| Signal-contract doc/Protocol | duck-typed, undocumented | a written contract + optional `typing.Protocol`; a test asserting strategies only read `≤ i` on a sentinel frame |
| Performance budget | none (no timeout) | a micro-benchmark/complexity smoke test (e.g. signal() must be ~O(1)/O(window) per call on a synthetic frame) |

## 13. Decision

**`AUDIT_PASS_WITH_PERFORMANCE_WARNINGS_VEORB_REJECT`.**
No validity/lookahead bug in this run; engine execution is causal; VE-ORB run
faithful and VE-ORB itself causal/timezone-safe. Real **MEDIUM systemic**
performance/fragility + test-coverage gaps exist. VE-ORB stays
**rejected/non-viable** — NO validation, NO holdout, NO demo/real/FTMO, NO
champion, NO portfolio candidate, NO incubation, NO edge.

## 14. Allowed Next Step

- **Archive/reject VE-ORB** (its zero-trades question is now fully closed:
  faithful run + causal engine + complete data + correct timezone ⇒ genuine
  degenerate selectivity; no further VE-ORB work).
- **Before large new strategy batches:** execute the lightweight, read-only/
  test-only contract-hardening phase described in
  `NEXT_PROMPT_ENGINE_CONTRACT_TESTS_OR_FIX_V1.md` (tests only; no engine/
  strategy behavior change without separate authorization; no backtest).

## 15. Forbidden Next Steps

NO VE-ORB revival/optimization. NO validation. NO holdout. NO 2025/2026. NO
optimization/sweep/walk-forward. NO engine/strategy/runner code change under
this audit. NO backtest / `--execute`. NO ZIP. NO `git add .`. NO declaring
edge/profitable/champion/demo/real/FTMO.

## 16. Findings Table

| id | severity | category | finding | evidence | implication | required_action |
|---|---|---|---|---|---|---|
| C1 | PASS | execution-causality | T+1 fill; signal@i, fill@i+1; exits per-bar ≤ i | engine.py:739 | No future-leak in harness | None |
| C2 | PASS | iteration | Full-period loop, no timeout/early-stop | engine.py:713 | VE-ORB silence is its own `None`; "engine cutoff" refuted | None |
| C3 | PASS | timezone | Frame index NY-localized (loader) + engine NY convert; DST-safe | data_loader.py:58–62/132/152; engine.py:636 | VE-ORB windows correct; timezone doubt closed | None |
| C4 | PASS | veorb-causality | VE-ORB reads only `:i`,`i-1`,`i`,backward roll,`iloc[i-200:i]` | strategy 170–223 | VE-ORB lookahead-safe | None |
| C5 | WARN | signal-contract | Engine passes **full multi-year frame**, no causal sandbox | engine.py:599,725 | Systemic lookahead surface for all strategies | Add universal no-lookahead test |
| C6 | WARN | contract-doc | `signal()` contract duck-typed, undocumented, no Protocol | runner.py:449–486; engine.py:721–725 | Easy to misuse in new strategies | Document contract / add Protocol+test |
| C7 | WARN | performance | VE-ORB O(N²): per-call full-frame ATR + O(i) OR scan | strategy 57–138; engine.py:713 | Batches can become infeasibly slow; uncaught | Add perf smoke test; (future) refactor pattern w/ authZ |
| C8 | WARN | fragility | `_infer_cadence_minutes` silently → all-`None` on edge cases | strategy 73–84 | Silent zero-trade strategies look "regime obsolete" | Add zero-activity sentinel |
| C9 | WARN | gate-blindspot | Reconciliation gate checks metric consistency, not signal density | runner.py:356–383 | 0-signal runs pass the gate cleanly | Add signal-density check before "obsolete" label |
| C10 | WARN | timeframe-traceability | Runner `target_timeframe="M1"` vs ~M5 frame / M15 summary / default `prepared_m5_bid` label | runner.py:454–457; engine.py:650 | Effective timeframe not pinned by artifacts | Pin & record actual strategy timeframe in manifest |
| C11 | WARN | repo-hygiene (out of scope) | preexisting unrelated dirty tree | `git status` | Not caused here; untouched | Repo owner only |
| — | BLOCKER | — | **NONE** | — | — | — |

## 17. Final Institutional Verdict

The engine/runner execution contract is causally sound and the VE-ORB run was
faithful: T+1 fills, full-period iteration with no timeout, NY-localized
DST-safe index, and a line-by-line-clean VE-ORB. Every residual technical doubt
from the prior diagnostic is now closed — VE-ORB's zero trades after 2015-02-02
are genuine degenerate selectivity, not an engine/timezone/cadence/data fault.
**Archive VE-ORB; do not spend more on it.** The real takeaway is systemic and
MEDIUM: the engine hands every strategy the entire multi-year frame with no
causal sandbox, no performance budget, and no zero-activity sentinel, and the
only no-lookahead test is TP01-specific. None of this broke this run, but the
next careless strategy could leak the future or silently die and still pass the
gate. Before any large new batch, run the lightweight, test-only
contract-hardening phase. No VE-ORB revival, no validation, no holdout, no edge.

---

*Read-only static contract audit. No code/data/runner/engine/strategy change,
no rerun, no validation/holdout/2025-2026, no optimization/sweep, no ZIP, no
heavy output committed, no `git add .`, no destructive git.*
