# TP01 FORMAL DOSSIER EXTERNAL AUDIT REPORT

**ROLE**: External institutional backtest auditor / metric-integrity auditor / risk manager / overfitting-prevention specialist / data-leakage auditor
**DATE**: 2026-05-17
**STRATEGY**: `tp01_london_ny_momentum_pullback`
**AUDITED BRANCH**: `research/tp01-formal-train-only-rerun-after-performance-fix-20260516`
**AUDITED COMMIT**: `66e42063d0b0f6ae8b4312374a254e5797c50021`
**REPORTED STATE UNDER REVIEW**: `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED`
**SCOPE**: Read-only artifact audit + recalculation from existing CSV ledgers. No new backtest, run, optimization, sweep, validation, holdout, news, high-precision, engine/strategy/data mutation.

---

## 1. Status

`TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`

The dossier cannot be accepted. Two independent, severe metric-integrity defects make every performance conclusion in the dossier untrustworthy, including the headline `+135.71% return` and the self-declared `SUCCESS_AND_SEALED` state.

---

## 2. Executive Summary

The dossier headlines **+135.71% total return / $235,710.51 ending equity / 1.32% max drawdown** while simultaneously reporting **Profit Factor 0.896 (<1)** and **expectancy −0.0684R (<0)**. These cannot coexist. Independent recalculation directly from the delivered trade ledgers confirms the strategy is **net-losing**, and uncovers a **second, separate defect**: a directional PnL **sign inversion** affecting ~49% of trades.

Two distinct defects were isolated:

1. **Equity curve decoupled from the trade ledger.** `equity_curve.csv` rises monotonically from $100,000.00 to $235,710.51, never dips below starting capital (min = $100,000.00), and its `drawdown_pct` column is **all zeros**. Yet the same run's `trades.csv` sums to **−13.06R / −$9,346.31** net. The reported `+135.71%` / `1.32% maxDD` are artifacts of this broken equity series, not real performance.

2. **Sign inversion in the trade ledger.** **94 of 191 trades (49.2%)** have a `pnl_r` sign opposite to the actual price direction. **63 trades exited via `stop_loss` are labeled `result=win`** and **17 trades exited via `take_profit` are labeled `result=loss`** — logically impossible. Example: 2015-01-05 `short`, entry 1.19104 → exit 1.192578875 (price rose against a short = loss), `exit_reason=stop_loss`, recorded as `result=win, pnl_r=+0.987`.

Because of defect (2), the strategy's *true* edge cannot be recovered by negation (R-distances are asymmetric: `target_rr=2.0`, stop buffer, forced closes, costs). Therefore **no profitability statement is possible** — the dossier is blocked, not merely "failed". The dossier's "100% mathematical equivalence to the pre-fix run" certification is vacuous: it certifies equivalence to an equally corrupted baseline.

Independently of the defects, every trustworthy view of the ledger-as-delivered is **net-negative** (Total R = −13.06, PF < 1, expectancy < 0) and **all 191 trades occur in 2015–2018 with zero activity 2019–2024**. TP-01 is not a viable first-wave candidate under any reading.

The defects reside in the **shared metric / equity-curve / drawdown layer**, so they contaminate **every strategy** processed by this harness, not only TP-01.

---

## 3. Files Audited

Light artifacts (tracked, read):

1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_FORMAL_TRAIN_ONLY_RERUN_AFTER_PERFORMANCE_FIX_REPORT.md`
2. `.../TP01_FORMAL_RERUN_20260516_212500/manifests/RUN_MANIFEST.json`
3. `.../TP01_FORMAL_RERUN_20260516_212500/configs/TP01_CONFIG_SNAPSHOT.json`
4. `.../TP01_FORMAL_RERUN_20260516_212500/reports/TP01_FORMAL_DOSSIER.md`
5. `.../TP01_FORMAL_RERUN_20260516_212500/tables/TP01_ANNUAL_SUMMARY.csv`
6. `.../TP01_FORMAL_RERUN_20260516_212500/tables/TP01_MONTHLY_SUMMARY.csv`
7. `.../TP01_FORMAL_RERUN_20260516_212500/tables/TP01_COST_PROFILE_SUMMARY.csv`
8. `.../TP01_FORMAL_RERUN_20260516_212500/tables/TP01_TRADE_DISTRIBUTION_SUMMARY.csv`

Heavy local artifacts (git-ignored, read only for recalculation, **not** committed):

- `local_outputs_do_not_commit/base/trades.csv` (191 trades), `base/equity_curve.csv` (728,996 rows)
- `local_outputs_do_not_commit/conservative/{trades,equity_curve}.csv`
- `local_outputs_do_not_commit/stress/{trades,equity_curve}.csv`

---

## 4. Metric Recalculation

Recomputed independently from `trades.csv` (and streamed `equity_curve.csv`). The recompute script lives **outside the repository** and is not committed.

### Base profile (191 trades)

| Metric | Recalculated (from ledger) | Dossier / Manifest | Verdict |
| :--- | :--- | :--- | :--- |
| Trades | 191 | 191 | MATCH |
| Wins / Losses (by `result` label) | 91 / 100 | — | — |
| Win rate | 47.643979% | 47.64397905759162% | MATCH |
| Sum `pnl_r` (Total R) | −13.0638682359 | −13.063868235890563 | MATCH |
| Sum `pnl_usd` | **−$9,346.31** | (not stated) | net loss |
| Gross win R / Gross loss R | 105.0557 / −118.1196 | — | — |
| Profit Factor (pnl_r partition) | 0.8894 | 0.8963989973230085 | <1 (minor method variance) |
| Expectancy R | −0.0683972159 | −0.068397215894715 | MATCH |
| Ending equity (additive: 100k + ΣUSD) | **$90,653.69 (−9.35%)** | $235,710.51 (+135.71%) | **CONTRADICTION** |
| Ending equity (compounded @0.5%R) | **$93,326.22 (−6.67%)** | $235,710.51 (+135.71%) | **CONTRADICTION** |
| Reconstructed max drawdown | **≈8.47%** (compounded) | 1.32% | **CONTRADICTION** |
| `equity_curve.csv` first / last / min / max | 100,000.00 / 235,710.51 / **100,000.00** / 235,710.51 | — | monotone, never < start |
| `equity_curve.csv` `drawdown_pct` column max | **0.000000** | — | column never populated |

Cross-check: `TP01_ANNUAL_SUMMARY.csv` sums to **ΣR = −13.06** and **ΣUSD = −$9,346.31** (2015 −$6,192.72; 2016 −$2,158.34; 2017 +$2,602.81; 2018 −$3,598.06); per-year `max_drawdown_pct` = 9.49 / 7.50 / 6.67 / 4.81%. An aggregate equity curve cannot have a 1.32% max drawdown when individual years draw down 4.8–9.5%.

### Conservative & Stress profiles (identical to each other)

| Metric | Recalculated | Manifest |
| :--- | :--- | :--- |
| Sum `pnl_r` | −13.7292934451 | — |
| Sum `pnl_usd` | −$10,253.59 | — |
| Profit Factor | 0.8815 | 0.8850167350442101 |
| Expectancy R | −0.0718811175 | −0.0718811175135023 |
| `equity_curve.csv` last / min | 235,405.41 / **100,000.00** | (ending $235,405.41) |
| Reported total return | — | +135.41% |

### Sign-inversion diagnostic (all profiles)

- **94 / 191 (49.2%)** trades: `pnl_r` sign ≠ realized price-direction sign.
- `result` × `exit_reason` cross-tab (base):

| result \ exit_reason | stop_loss | take_profit | forced_session_close |
| :--- | :---: | :---: | :---: |
| **win** | **63 (impossible)** | 19 | 9 |
| **loss** | 68 | **17 (impossible)** | 15 |

A `stop_loss` exit is by definition a loss; a `take_profit` exit is by definition a win. 63 + 17 = 80 logically impossible labels, plus 14 ambiguous forced-close labels, consistent with an inverted directional PnL computation (long/short sign handling).

---

## 5. PF / Expectancy / Equity Consistency

**Internally consistent (describe a losing system):** Total R (−13.06), expectancy (−0.0684R), PF (<1), win rate (47.64%), Σ`pnl_usd` (−$9,346.31), and the per-year/per-month tables. These all agree: **the ledger-as-delivered is net-negative.**

**Mutually impossible:** `total_return_pct` (+135.71%), `ending_equity` ($235,710.51), and aggregate `max_dd_pct` (1.32%) versus the above. Root cause is **not** sign inversion at the equity level (sign inversion would have flipped PF too; PF/expectancy are *consistently* negative and track cost profiles correctly: 0.8964 base vs 0.8850 conservative/stress). The cause is an **equity-curve / total-return / drawdown computation that is decoupled from the trade ledger**:

- `equity_curve.csv` is monotonically non-decreasing, `min == start == $100,000.00`, ends at exactly the reported figure, `drawdown_pct` column is entirely `0.0`.
- The reported `1.32% maxDD` is merely the largest intra-step wiggle of that broken monotone series, not a real drawdown.
- All three cost profiles produce ≈+135.5% despite correctly-differentiated PF/expectancy → the return/equity figure is **not derived from the trades it claims to summarize**.

A **second, independent defect** is the **sign inversion** inside the trade ledger itself (Section 4). Two separate bugs, both in the shared metric layer.

**Conclusion:** Equity, total return, and max drawdown are **fabricated artifacts of a broken aggregation layer**. The trade-level edge metrics are real but corrupted by sign inversion. Nothing in the dossier supports a profitability conclusion.

---

## 6. Annual Activity Audit

Confirmed directly from `trades.csv` `entry_time_ny` (all three profiles identical):

| Year | Trades | Source confirmation |
| :--- | :---: | :--- |
| 2015 | 57 | ledger + ANNUAL_SUMMARY |
| 2016 | 66 | ledger + ANNUAL_SUMMARY |
| 2017 | 63 | ledger + ANNUAL_SUMMARY |
| 2018 | 5 | ledger + ANNUAL_SUMMARY |
| 2019–2024 | 0 | no rows present in ledger |
| **Total** | **191** | MATCH |

`TP01_TRADE_DISTRIBUTION_SUMMARY.csv` independently corroborates: Days with 0 trades = 2,931; entry hours confined to NY 08–11h; `MONTHLY_SUMMARY` ends at `2018-01`. The 2019–2024 inactivity is a **genuine property of the run output** (not an artifact of the metric bug), but its *cause* (regime vs filter vs loader/timezone) is unverifiable from artifacts alone — see Section 7.

---

## 7. Regime Drift Assessment

The dossier attributes the post-2018 silence to "structural volatility compression" defeating the ATR-50th-percentile + `momentum ≥ 1.5×ATR` filter inside the narrow 08:00–12:00 NY window. Auditor position:

- **Mechanically plausible**: a strict rolling-percentile volatility gate plus a 4-hour window can legitimately go quiet for years.
- **Not independently verified**: confirming regime drift vs. an over-restrictive filter vs. a **session/timezone or data-loader defect** would require re-running diagnostics / the engine, which is **out of audit scope** (forbidden). The dossier's `check_loaded_frame.py` claim of full 2018–2024 data population is asserted, not reproduced here.
- **Moot for the decision**: even if regime drift is real, the run's metric layer is corrupted, so the dossier cannot be accepted regardless. Classification of the activity finding: **unresolved / secondary to the metric-integrity block**, with `FORMAL_TRAIN_FAIL_REGIME_OBSOLESCENCE` as a strong *secondary* characterization (a 2015–2018-only strategy is obsolete for forward use even after metric remediation).

---

## 8. Cost Sensitivity

| Profile | PF | Expectancy R | Σ pnl_usd | Reported total return |
| :--- | :---: | :---: | :---: | :---: |
| base | 0.8894 (rep. 0.8964) | −0.0684 | −$9,346.31 | +135.71% (corrupt) |
| conservative | 0.8815 (rep. 0.8850) | −0.0719 | −$10,253.59 | +135.41% (corrupt) |
| stress | 0.8815 (rep. 0.8850) | −0.0719 | −$10,253.59 | +135.41% (corrupt) |

Cost behaviour is *directionally* sane on the trade metrics: heavier costs deepen the loss (PF and expectancy degrade monotonically). **Conservative and stress are byte-identical** — the stress multipliers (`stress_spread_multiplier=1.35`, `stress_slippage_multiplier=1.6`) did **not** differentiate the stress run from conservative; this is a **third anomaly** worth flagging (stress profile may not be exercising its multipliers). The `total_return_pct` column is the corrupt equity figure across all three profiles and carries no information.

---

## 9. Safety / Leakage Audit

| Control | Evidence | Result |
| :--- | :--- | :--- |
| No holdout | `RUN_MANIFEST.holdout_used=false`; `CONFIG.no_holdout=true`; report §5 attests `sealed_holdout_2025_2026/` never opened | PASS (per artifacts) |
| No 2025/2026 | `trades.csv` entries only 2015–2018; `equity_curve.csv` last ts `2024-12-31 17:00:00`; data_range `2015-01-01..2024-12-31` | PASS |
| No optimization | `RUN_MANIFEST.optimization_run=false` | PASS |
| No sweep / validation | No sweep/validation artifacts; single param set in `CONFIG_SNAPSHOT` | PASS |
| No news | report §5 `NewsConfig(enabled=False)`; no news files referenced | PASS |
| No high precision | `CONFIG price_source='bid'`, `execution_mode='normal_mode'` (not bid/ask high-precision) | PASS |
| No data mutation | Auditor only read CSVs; no writes to data vault | PASS |
| No engine/strategy mutation | Auditor only read artifacts + recomputed from CSV | PASS |
| Auditor performed no new run | No backtest/sweep/opt executed by auditor | PASS |

**Caveats (not leakage, but integrity):** the dossier's "100% mathematical equivalence" and "indicator caching is causal / no lookahead" claims are **not independently verifiable** from artifacts and are **moot**, since the metrics they certify are themselves corrupted (Sections 4–5). No positive evidence of holdout/2025-26/news leakage was found; the **failure is metric integrity, not data leakage**.

---

## 10. Git / Output Policy Audit

| Check | Result |
| :--- | :--- |
| Heavy outputs tracked? | **No** — `git ls-files` shows zero `local_outputs_do_not_commit/*`; `git check-ignore` confirms they are ignored |
| Run folder committed surface | Light only: `configs/*.json`, `manifests/*.json`, `profile_reports/*.{csv,json}`, `reports/*.md`, `tables/*.csv`. No heavy CSV, no ZIP, no binaries |
| Root files added | None |
| Engine / strategy code touched | None |
| Data vault touched | None |
| Pre-existing unrelated changes | Modified/untracked files exist only under `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/` — **left untouched; explicitly NOT staged** |
| This audit's commit surface | Docs only (3 markdown files in `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`), staged explicitly (no `git add .`) |

Output/commit policy: **COMPLIANT.**

---

## 11. Decision

**`TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`** — the dossier is **rejected**.

- The self-declared `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED` state is **overturned**.
- The headline **`+135.71% return / $235,710.51 / 1.32% maxDD` is rejected** as a corrupt artifact of a decoupled equity/drawdown layer.
- The trade ledger is additionally compromised by a **49% sign inversion**; the strategy's true edge **cannot be stated**.
- The "mathematical equivalence" certification is **void** (equivalence to a corrupted baseline proves nothing).
- Robust, defect-independent facts: ledger-as-delivered is **net-negative** (−13.06R, PF < 1, expectancy −0.068R) and **active only 2015–2018** (0 trades 2019–2024).

No champion designation. No "profitable" designation. No promotion to incubation. No portfolio candidacy.

---

## 12. Recommendation

TP-01 (`tp01_london_ny_momentum_pullback`):

- **REJECTED — removed from the first wave.** Not watchlist-as-promising, not a portfolio candidate, not incubation, not champion.
- **Requires metric-engine remediation** (two bugs: decoupled equity/DD aggregation; directional PnL sign inversion) **before any re-evaluation of any strategy**, because the defects are in the **shared** metric/equity layer and contaminate every strategy run by this harness.
- **Requires strategy redesign / regime re-justification** independently: even with correct metrics, a strategy that produced 0 trades for 6 of 10 years (2019–2024) is **regime-obsolete** for forward deployment and would not pass a train-only viability bar on the delivered (net-negative) ledger.
- Treat all prior dossiers produced by this harness (any strategy) as **suspect** until the metric layer is fixed and re-verified against hand-computed trade ledgers.

---

## 13. Next Step

Because TP-01 fails **and** a metric inconsistency is present, both follow-up prompts are created (per audit protocol):

1. **PRIMARY — `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_FIX_TP01_METRIC_INCONSISTENCY.md`**
   Remediate the shared metric layer: (a) rebuild `equity_curve` / `total_return` / `max_drawdown` so they are derived from the trade ledger; (b) fix the directional `pnl_r` / `result` sign for short trades (and the `stop_loss→win` / `take_profit→loss` labeling); (c) investigate why `stress` == `conservative`. Gate: no strategy dossier may be sealed until recomputed metrics reconcile to an independent hand calculation.

2. **SEQUENCED — `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_FORMAL_TRAIN_ONLY_MR01_BACKTEST.md`**
   Proceed to the next queued strategy (MR-01) formal train-only backtest **only after** the metric fix above is merged and re-verified. Running MR-01 on the current harness would inherit the identical corruption and is therefore **blocked until the fix lands**.

---
*External audit complete. Dossier blocked on metric integrity. No new runs, optimizations, or data/engine changes were performed during this audit.*
