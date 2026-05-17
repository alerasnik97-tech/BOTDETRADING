# TP01 FORMAL DOSSIER EXTERNAL AUDIT REPORT

**AUDIT TIER**: External institutional, conservative, no-concessions (v2 hardened protocol)
**ROLE COMPOSITE**: Quant backtesting auditor · metric-integrity auditor · risk manager · FX systematic reviewer · overfitting-prevention specialist · data-leakage auditor · git/output-hygiene auditor · pre-portfolio gatekeeper
**DATE**: 2026-05-17
**STRATEGY**: `tp01_london_ny_momentum_pullback`
**BASE BRANCH**: `research/tp01-formal-train-only-rerun-after-performance-fix-20260516`
**AUDITED COMMIT**: `66e42063d0b0f6ae8b4312374a254e5797c50021` (origin tip; confirmed via `git fetch --prune`)
**AUDIT BRANCH**: `audit/tp01-formal-dossier-external-audit-v2-20260517` (v1 branch already existed → v2 per protocol)
**REPORTED STATE UNDER REVIEW**: `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED`
**SCOPE**: read-only artifact audit + recalculation from existing CSV ledgers. No new run/backtest/opt/sweep/validation/holdout/news/high-precision; no engine/strategy/data mutation.

---

## 1. Status

`TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`

The dossier is **rejected and blocked**. Four independent integrity defects (one new vs v1) make every performance and cost conclusion untrustworthy. The self-declared `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED` state is overturned. Narrative ("100% equivalence", "data fully populated 2018–2024") is **not accepted as evidence**.

---

## 2. Executive Summary

The dossier headlines **+135.71% return / $235,710.51 ending equity / 1.32% maxDD** while simultaneously reporting **PF 0.896 (<1)** and **expectancy −0.0684R (<0)**, and the engine's own `summary.json` reports **3 negative years / 1 positive year / 21–22 negative months**. A system that loses in 3 of 4 traded years with PF<1 cannot return +135%. Independent recalculation from the delivered ledgers confirms the run is **net-losing** and surfaces four defects:

1. **Equity curve decoupled from the trade ledger.** `equity_curve.csv` rises from $100,000.00 to $235,710.51, its minimum never drops below the $100,000 start, it has only 272 trivial down-ticks, and its `drawdown_pct` column is **all zeros** — while the same run's `trades.csv` sums to **−13.06R / −$9,346.31**. The +135.71% / 1.32% maxDD figures are artifacts of this broken series.
2. **Directional sign inversion in the trade ledger.** **94 / 191 trades (49.2%)** have `pnl_r` sign opposite to realized price direction. **63 `stop_loss` exits are labeled `result=win`**; **17 `take_profit` exits are labeled `result=loss`** — logically impossible.
3. **Engine-native `summary.json` is internally self-contradictory.** Per-profile `summary.json` reports PF<1, expectancy<0, 3 negative years, 1 positive year, 21–22 negative months — yet `total_return_pct ≈ +135%` and `max_drawdown_pct ≈ 1.3%`. The defect is in the **core metric layer**, not only the dossier markdown.
4. **NEW — Cost-profile mislabeling/duplication.** `base/summary.json` = `{profile:base, normal_mode}`. **Both** `conservative/summary.json` **and** `stress/summary.json` = `{profile:stress, conservative_mode}`. Only **two** distinct runs exist (base + one stress-tagged/conservative-mode run duplicated into the conservative *and* stress folders). The manifest's claim of "3 cost profiles run" is **false**: there is no genuine conservative profile and the stress profile was not executed in a stress mode.

Because of defect (2) the strategy's true edge cannot be recovered by negation (asymmetric R: `target_rr=2.0`, stop buffer, forced closes, costs), so **no profitability statement is possible** — the dossier is blocked, not merely "low-edge fail". All defects reside in the **shared engine metric/cost layer** → they contaminate **every strategy** processed by this harness.

Defect-independent facts (robust regardless of the bugs): the ledger-as-delivered is **net-negative** (Total R −13.06, PF<1, expectancy −0.068R, median R ≈ −0.55) and **all 191 trades fall in 2015–2018 with zero activity 2019–2024**. TP-01 is not a first-wave candidate under any reading.

---

## 3. Files Audited

Light, tracked (read):

1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_FORMAL_TRAIN_ONLY_RERUN_AFTER_PERFORMANCE_FIX_REPORT.md`
2. `…/TP01_FORMAL_RERUN_20260516_212500/manifests/RUN_MANIFEST.json`
3. `…/configs/TP01_CONFIG_SNAPSHOT.json`
4. `…/reports/TP01_FORMAL_DOSSIER.md`
5. `…/tables/TP01_ANNUAL_SUMMARY.csv`
6. `…/tables/TP01_MONTHLY_SUMMARY.csv`
7. `…/tables/TP01_COST_PROFILE_SUMMARY.csv`
8. `…/tables/TP01_TRADE_DISTRIBUTION_SUMMARY.csv`
9. (extra) `…/profile_reports/{base,conservative,stress}/summary.json`, `optimization_results.csv`

Heavy, git-ignored (read only for recalculation, **not committed**):

- `local_outputs_do_not_commit/{base,conservative,stress}/trades.csv` (191 each)
- `local_outputs_do_not_commit/{base,conservative,stress}/equity_curve.csv` (728,996 rows each)

Recompute script lives **outside the repository** (working-dir parent) and is not committed.

---

## 4. Commit Surface Audit

| Commit | Message | Surface |
| :--- | :--- | :--- |
| `66e42063` | gov: added formal train-only rerun audit and closure report | **1 file** — `TP01_FORMAL_TRAIN_ONLY_RERUN_AFTER_PERFORMANCE_FIX_REPORT.md` (pure doc) |
| `18a507fe` | gov: completed formal train-only rerun … 3 cost profiles | 19 light files: `*.json` (configs/manifests/summary), `*_stats.csv`, `*.md` dossier, `TP01_*` tables |

Scan for `\.zip$ / \.py$ / engine / strategy code / trades.csv / equity_curve.csv / local_outputs / parquet / h5 / root files` across both commits → **none**. `optimization_results.csv` (all 3 profiles) = **2 bytes (empty)** → benign template placeholder, consistent with `optimization_run=false`. Heavy CSVs are git-ignored and untracked (`git check-ignore` confirms). **Output policy: COMPLIANT** (not `BLOCKED_OUTPUT_POLICY`).

---

## 5. Metric Recalculation

Independent recompute from `trades.csv` / streamed `equity_curve.csv`. RISK = 0.5% per trade (`risk_pct=0.5`).

### Base (191 trades) — `normal_mode`, profile `base`

| Metric | Recalculated | Dossier / summary.json | Verdict |
| :--- | :--- | :--- | :--- |
| Trades / wins(label) / losses(label) | 191 / 91 / 100 | 191 | MATCH |
| Win rate | 47.643979% | 47.64397905759162% | MATCH |
| Gross win R / Gross loss R | 105.0557 / −118.1196 | — | — |
| Profit factor | 0.8894 (pnl_r partition) | 0.8963989973230085 | <1 (minor method variance) |
| Total R | −13.0638682359 | −13.063868235890563 | MATCH |
| Expectancy R / Avg R | −0.0683972159 | −0.068397215894715 | MATCH |
| Median R | **−0.5538** | — | strongly negative |
| Avg win R / Avg loss R | 1.1545 / −1.1812 | — | payoff < 1 |
| Σ pnl_usd | **−$9,346.31** | — | net loss |
| Ending equity (additive) | **$90,653.69 (−9.35%)** | $235,710.51 (+135.71%) | **CONTRADICTION** |
| Ending equity (compounded@0.5%R) | **$93,326.22 (−6.67%)** | $235,710.51 | **CONTRADICTION** |
| Reconstructed max DD | **≈8.47%** | 1.32% | **CONTRADICTION** |
| Annual USD | 2015 −6,192.72 · 2016 −2,158.34 · 2017 +2,602.81 · 2018 −3,598.06 | — | 3 neg / 1 pos yr |
| `equity_curve.csv` first/last/min/max | 100,000.00 / 235,710.51 / **100,000.00** / 235,710.51 | — | never < start |
| `equity_curve.csv` decreasing steps | **272** (of 728,996) | — | quasi-monotone up |
| `equity_curve.csv` `drawdown_pct` col max | **0.00000000** | — | column never populated |

### Conservative & Stress (byte-identical to each other)

| Metric | Recalculated | summary.json |
| :--- | :--- | :--- |
| Total R | −13.7292934451 | — |
| Σ pnl_usd | −$10,253.59 | — |
| Profit factor | 0.8815 (pnl_r) | 0.8850167350442101 |
| Expectancy R | −0.0718811175 | −0.0718811175135023 |
| Median R | −0.5159 | — |
| Annual USD | 2015 −5,547.26 · 2016 −3,957.09 · 2017 +2,807.06 · 2018 −3,556.30 | — |
| Ending equity (additive) | **$89,746.41 (−10.25%)** | $235,405.41 (+135.41%) | **CONTRADICTION** |
| `equity_curve.csv` last / min | 235,405.41 / **100,000.00** | — |
| `summary.json` `costs_used` | — | **{execution_mode: conservative_mode, cost_profile: stress}** (both folders) |

Cross-check vs `TP01_ANNUAL_SUMMARY.csv`: ΣR = −13.06, ΣUSD = −$9,346.31, per-year DD 9.49/7.50/6.67/4.81% — an aggregate equity curve cannot show 1.32% maxDD when individual years draw down 4.8–9.5%.

---

## 6. PF / Expectancy / Equity Consistency

**Internally consistent (all describe a losing system):** Total R, expectancy, PF, win rate, median R, Σ pnl_usd, per-year/per-month tables, `negative_years=3 / positive_years=1`, `negative_months=21–22`.

**Mutually impossible:** `total_return_pct ≈ +135%`, `ending_equity ≈ $235k`, `max_dd_pct ≈ 1.3%` vs all of the above.

**Root cause is NOT equity-level sign flip** (that would flip PF too; PF/expectancy are consistently negative and track cost differences). It is an **equity/return/drawdown aggregation decoupled from the trade ledger**: `equity_curve.csv` rises while trades lose; `min == start`; `drawdown_pct` all zeros; the same corruption appears in engine-native `summary.json`. A **second, independent defect** is the **sign inversion inside the ledger** (Section 5/8). Both bugs are in the shared engine metric layer. **Reconciliation: FAILED.** → `TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`.

---

## 7. Annual Activity Audit

Confirmed from `trades.csv` `entry_time_ny` (all profiles identical): **191 total** — 2015 **57** · 2016 **66** · 2017 **63** · 2018 **5** · 2019–2024 **0**. 37 active months, first `2015-01`, last `2018-01`. Independently corroborated by `TP01_ANNUAL_SUMMARY.csv`, `MONTHLY_SUMMARY` (ends 2018-01), and `TRADE_DISTRIBUTION_SUMMARY` (Days with 0 trades = 2,931). The 2019–2024 silence is a genuine property of the output.

---

## 8. Post-2018 Inactivity Assessment

The dossier attributes the silence to volatility-regime compression defeating the ATR-50th-pct + `momentum ≥ 1.5×ATR` filter inside 08:00–12:00 NY.

- The dossier's supporting checks (`check_loaded_frame.py` data completeness; "no NaN/inf 2018–2024"; indicators populated) are **asserted, not reproduced** in the artifacts. Per "no aceptar reporte por narrativa", these are **not accepted as evidence**.
- Distinguishing *regime drift* vs *over-restrictive filter* vs *timezone/loader/cache/condition bug* would require diagnostic re-execution / engine inspection — **forbidden in scope**.
- Activity classification: **`BLOCKED_ACTIVITY_DIAGNOSIS_INSUFFICIENT`** (cause cannot be certified within scope). Practical risk read for the gatekeeper: **`STRATEGY_OBSOLETE_IN_RECENT_REGIME`** — a strategy that produced 0 trades in 6 of 10 years is not forward-deployable regardless of cause. Either way it does not unblock the dossier.

---

## 9. Cost Profile Audit

| Profile (folder) | summary.json `cost_profile` | `execution_mode` | Distinct run? |
| :--- | :--- | :--- | :--- |
| base | base | normal_mode | YES |
| conservative | **stress** | **conservative_mode** | NO — duplicate of stress |
| stress | stress | conservative_mode | NO — same as conservative |

- Base vs conservative/stress trade ledgers **do differ** (per-year USD differs; cost model is partially functional base↔non-base).
- **conservative and stress are byte-identical** in trades, equity, PF, expectancy, every year — and **both folders' `summary.json` self-report `cost_profile=stress, execution_mode=conservative_mode`**. There is **no genuine conservative-cost run**, and the **stress profile was never executed in a stress mode**. `RUN_MANIFEST.json`'s `"cost_profiles_run": ["base","conservative","stress"]` and its three distinct result rows are **misleading**: only two runs exist; the conservative and stress manifest rows carry identical numbers.
- Classification: **`COST_PROFILE_APPLICATION_BLOCKED`** (base valid; conservative non-existent; stress mislabeled and mis-executed). Cost sensitivity to a true stress scenario is **unknown**.

---

## 10. Safety / Leakage Audit

Token scan (`2025|2026|holdout|sealed_holdout|validation|forex_factory|news|high_precision|level2|optimization|sweep|walk_forward|incubation|production`) over the rerun report + run folder + manifest; hits classified:

| Hit | Location | Class |
| :--- | :--- | :--- |
| "zero holdout leakage / zero 2025/2026 / no news / no high-precision" | report L13 | negative declaration |
| "sealed_holdout_2025_2026 was never opened" | report L88 | negative declaration |
| "NewsConfig(enabled=False), news unreferenced" | report L89 | negative declaration |
| `holdout_used:false`, `optimization_run:false` | RUN_MANIFEST | negative declaration |
| "100% equivalence", "no NaN/inf 2018–2024" | report L39/L66 | unverified narrative (not evidence) |
| `2026-05-1x` timestamps, branch name, run_id | report/manifest | benign metadata |

**No blocker-class hits.** Independent confirmation: `trades.csv` entries are 2015–2018 only; `equity_curve.csv` last datetime `2024-12-31 17:00:00`; data_range `2015-01-01..2024-12-31`. `optimization_results.csv` empty (all profiles).

| Control | Verdict |
| :--- | :--- |
| holdout_used | **false** (manifest + config `no_holdout=true`) |
| 2025/2026 used | **false** (no rows beyond 2024-12-31 in trades/equity) |
| validation_run | **false** (no artifacts) |
| optimization_run | **false** (manifest; empty optimization_results.csv) |
| sweep_run | **false** (single param set) |
| news_used | **false** (`NewsConfig(enabled=False)`, `news_filter_used:false`) |
| high_precision_used | **false** (`price_source='bid'`, not bid/ask HP) |
| engine_modified (by audit) | **false** |
| data_modified (by audit) | **false** |
| new run by auditor | **false** |

Leakage posture: **PASS on data leakage** (no positive evidence of holdout/2025-26/news/opt/sweep). The failure is **metric & cost-label integrity**, not leakage. Equivalence/data-completeness claims rejected as narrative.

---

## 11. Final Classification

`TP01_BLOCKED_METRIC_INCONSISTENCY`

Rationale: equity decoupled from ledger + 49% directional sign inversion + self-contradictory engine `summary.json` + cost-profile mislabel/duplication. The inputs are corrupted, so neither `REJECTED_LOW_EDGE` nor `REJECTED_REGIME_OBSOLESCENCE` can be issued as a *final* verdict (they assume trustworthy metrics). BLOCKED supersedes. Not champion, not portfolio candidate, not eligible for validation/holdout.

---

## 12. Decision

- **Dossier REJECTED / BLOCKED.** `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED` is **overturned**; the `+135.71%` headline, `1.32% maxDD`, and "3 cost profiles" claims are **rejected as corrupt artifacts**.
- The "100% mathematical equivalence" certification is **void** — it certifies equivalence to an equally corrupted baseline and is unverifiable from artifacts.
- **TP-01 cannot advance.** No champion, no rentability statement, no incubation, no FTMO/demo/real, no portfolio candidacy, no validation/holdout.
- **MR-01 formal cannot proceed yet** — the defects are in the shared engine metric/cost layer; any new strategy run inherits identical corruption.
- All prior dossiers from this harness must be treated as **suspect** until the metric/cost layer is fixed and reconciled against hand-computed ledgers.

---

## 13. Recommendation

- TP-01: **BLOCKED on metric integrity** + (defect-independent) **negative edge** (PF<1, expectancy −0.068R, median R −0.55, 3 neg/1 pos year) + **2019–2024 inactivity** ⇒ **remove from first wave**; **requires metric-engine remediation AND strategy/regime redesign** before any re-evaluation. Not watchlist-as-promising; not a portfolio candidate.
- Priority is the **shared metric/cost-layer fix**, not TP-01 itself. Until fixed, **no strategy dossier may be sealed** and **MR-01 formal is gated**.
- Add a mandatory **ledger↔dossier reconciliation gate** (independent recompute must match within strict tolerance; sign invariants `stop_loss⇒loss`, `take_profit⇒win`; equity must derive from the ledger; each cost profile must self-report its own profile/mode) before any future `*_SUCCESS_AND_SEALED`.

---

## 14. Copy-Paste Summary for ChatGPT

```
TP-01 formal train-only dossier (2015–2024) EXTERNAL AUDIT v2 → STATUS: TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY.
Audited commit 66e42063 on branch research/tp01-formal-train-only-rerun-after-performance-fix-20260516; audit branch audit/tp01-formal-dossier-external-audit-v2-20260517.
4 integrity defects (shared engine layer): (1) equity_curve decoupled from ledger — rises to $235,710.51/+135.71% with drawdown_pct all zeros and min never below $100k, while trades.csv sums to -13.06R / -$9,346.31; (2) directional sign inversion in 94/191 trades (49.2%) — 63 stop_loss labeled win, 17 take_profit labeled loss; (3) engine-native summary.json self-contradictory (PF<1, exp<0, 3 negative years / 1 positive, 21–22 negative months, yet +135% return); (4) cost-profile mislabel — base=base/normal_mode, but BOTH conservative/ and stress/ summary.json say cost_profile=stress/conservative_mode → only 2 distinct runs, no genuine conservative, stress never run in stress mode (manifest "3 profiles" is false).
Recalculated (base): trades 191, win rate 47.64%, PF 0.889 (<1), expectancy -0.0684R, total R -13.06, median R -0.55, true ending equity ≈ $90,654 (-9.35% additive) / $93,326 (-6.67% compounded), real maxDD ≈ 8.5%. Activity: 2015:57, 2016:66, 2017:63, 2018:5, 2019–2024:0.
Safety: no data leakage (no holdout/2025-26/news/opt/sweep/high-precision; trades 2015–2018 only); equivalence & data-completeness claims rejected as unverified narrative. Git/output policy: COMPLIANT (docs/light only, heavy CSV git-ignored).
Decision: dossier overturned; TP-01 removed from first wave; NOT champion/portfolio/incubation. MR-01 formal GATED behind a shared metric/cost-layer fix + a mandatory ledger↔dossier reconciliation gate.
Next: NEXT_PROMPT_FIX_TP01_METRIC_INCONSISTENCY.md (primary); NEXT_PROMPT_FORMAL_TRAIN_ONLY_MR01_BACKTEST.md (blocked until fix lands).
```

---
*External audit v2 complete. Dossier blocked on metric & cost-label integrity. No new runs, optimizations, validations, or data/engine changes were performed during this audit.*
