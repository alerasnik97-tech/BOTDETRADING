# TP01 FORMAL DOSSIER EXTERNAL AUDIT REPORT

**AUDIT TIER**: External institutional, conservative, no-concessions — pass v3 (third independent reproduction)
**ROLE COMPOSITE**: Quant backtest auditor · metric-integrity auditor · risk manager · data-leakage auditor · overfitting-prevention specialist · pre-portfolio gatekeeper
**DATE**: 2026-05-17
**STRATEGY**: `tp01_london_ny_momentum_pullback`
**BASE BRANCH**: `research/tp01-formal-train-only-rerun-after-performance-fix-20260516`
**AUDITED COMMIT**: `66e42063d0b0f6ae8b4312374a254e5797c50021` (origin tip; confirmed via `git fetch --prune`, unchanged)
**AUDIT BRANCH**: `audit/tp01-formal-dossier-external-audit-v3-20260517` (v1 + v2 already exist on origin → v3 per no-collision/no-force rule)
**REPORTED STATE UNDER REVIEW**: `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED`
**SCOPE**: read-only artifact audit + recalculation from existing ledgers. No new run/backtest/opt/sweep/validation/holdout/news/high-precision; no engine/strategy/data mutation.
**REPRODUCIBILITY**: decisive metrics independently recomputed in 3 separate passes on the same commit — **identical results each time**. The inconsistency is systematic and deterministic, not stochastic noise.

---

## 1. Status

`TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`

Dossier **rejected and blocked**. Four independent integrity defects make every performance/cost conclusion untrustworthy. The self-declared `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED` is overturned. Narrative claims ("100% equivalence", "data fully populated 2018–2024") are **not accepted as evidence**.

---

## 2. Executive Summary

The dossier headlines **+135.71% / $235,710.51 / 1.32% maxDD** while reporting **PF 0.896 (<1)** and **expectancy −0.0684R (<0)**; engine-native `summary.json` reports **3 negative years / 1 positive / 21–22 negative months**. A PF<1 system losing in 3 of 4 traded years cannot return +135%. Independent recompute (×3) confirms a **net-losing** run and four defects:

1. **Equity curve decoupled from the ledger.** `equity_curve.csv` ends at `235710.51` (base) / `235405.41` (cons/stress), its minimum never drops below the $100,000 start, only 272 trivial down-ticks in 728,996 rows, `drawdown_pct` column entirely `0.0` — while `trades.csv` sums to **−13.06R / −$9,346.31** (base). The +135% / 1.3% maxDD are artifacts of this broken series.
2. **Directional sign inversion.** **94/191 trades (49.2%)** have `pnl_r` sign opposite to realized price direction. **63 `stop_loss` exits labeled `result=win`; 17 `take_profit` exits labeled `result=loss`** — logically impossible.
3. **Engine-native `summary.json` self-contradictory.** Per-profile `summary.json`: PF<1, expectancy<0, 3 neg / 1 pos year, 21–22 neg months — yet `total_return_pct≈+135%`, `max_drawdown_pct≈1.3%`. Defect is in the **core metric layer**, not just the markdown.
4. **Cost-profile mislabel/duplication.** `base/summary.json` = `{base, normal_mode}`. **Both** `conservative/summary.json` **and** `stress/summary.json` = `{cost_profile:stress, execution_mode:conservative_mode}`; the `conservative/` and `stress/` `equity_curve.csv` are byte-identical (34,209,161 bytes). Only **2** distinct runs exist (no genuine conservative; stress never run in stress mode). `RUN_MANIFEST`'s "3 cost profiles" is **false**.

Because of (2), true edge cannot be recovered by negation (asymmetric R: `target_rr=2.0`, stop buffer, forced closes, costs) → **no profitability statement possible**; BLOCKED, not "low-edge fail". All defects are in the **shared engine metric/cost layer** → they contaminate **every strategy** on this harness.

Defect-independent facts: ledger-as-delivered is **net-negative** (Total R −13.06, PF<1, expectancy −0.068R, median R −0.55) and **all 191 trades fall in 2015–2018; zero 2019–2024**. TP-01 is not a first-wave candidate under any reading.

---

## 3. Files Audited

Light/tracked: (1) `…/lab_readiness/TP01_FORMAL_TRAIN_ONLY_RERUN_AFTER_PERFORMANCE_FIX_REPORT.md`; (2) `…/manifests/RUN_MANIFEST.json`; (3) `…/configs/TP01_CONFIG_SNAPSHOT.json`; (4) `…/reports/TP01_FORMAL_DOSSIER.md`; (5) `…/tables/TP01_ANNUAL_SUMMARY.csv`; (6) `…/tables/TP01_MONTHLY_SUMMARY.csv`; (7) `…/tables/TP01_COST_PROFILE_SUMMARY.csv`; (8) `…/tables/TP01_TRADE_DISTRIBUTION_SUMMARY.csv`; (extra) `…/profile_reports/{base,conservative,stress}/summary.json` + `optimization_results.csv`.

Heavy/git-ignored (read only for recalc, **not committed**): `local_outputs_do_not_commit/{base,conservative,stress}/{trades.csv (191), equity_curve.csv (728,996 rows)}`. Recompute script kept **outside the repo** and deleted after use.

---

## 4. Commit Surface Audit

| Commit | Message | Surface |
| :--- | :--- | :--- |
| `66e42063` | gov: added formal train-only rerun audit and closure report | **1 file** — `TP01_FORMAL_TRAIN_ONLY_RERUN_AFTER_PERFORMANCE_FIX_REPORT.md` (doc) |
| `18a507fe` | gov: completed formal train-only rerun … 3 cost profiles | 19 light files (`*.json`, `*_stats.csv`, dossier `*.md`, `TP01_*` tables) |

Scan across both commits for `.zip / .py / engine / strategies / local_outputs / trades.csv / equity_curve / parquet / h5 / root files` → **none**. `optimization_results.csv` (all profiles) = empty (consistent with `optimization_run=false`). Heavy CSVs git-ignored & untracked. **Output policy: COMPLIANT** (not `BLOCKED_OUTPUT_POLICY`).

---

## 5. Metric Recalculation

Independent recompute from `trades.csv`; RISK=0.5%/trade. Results identical across audit passes v1/v2/v3.

### Base (191) — `normal_mode`/`base`

| Metric | Recalculated | Reported | Verdict |
| :--- | :--- | :--- | :--- |
| Trades / wins / losses | 191 / 91 / 100 | 191 | MATCH |
| Win rate | 47.643979% | 47.64397905759162% | MATCH |
| Gross win R / loss R | 105.0557 / −118.1196 | — | — |
| Profit factor | 0.8894 (pnl_r) | 0.8964 | <1 |
| Total R | −13.0638682359 | −13.063868235890563 | MATCH |
| Expectancy R / Avg R | −0.0683972159 | −0.068397215894715 | MATCH |
| Median R | **−0.5539** | — | strongly negative |
| Σ pnl_usd | **−$9,346.31** | — | net loss |
| Ending equity (additive) | **$90,653.69 (−9.35%)** | $235,710.51 (+135.71%) | **CONTRADICTION** |
| Ending equity (compounded@0.5%R) | **≈$93,326 (−6.67%)** | $235,710.51 | **CONTRADICTION** |
| Reconstructed max DD | **≈8.5%** | 1.32% | **CONTRADICTION** |
| `equity_curve.csv` head / tail | `100000.0` / **`235710.51000547045`** | — | rises while ledger loses |
| `equity_curve.csv` min / decreasing steps / `drawdown_pct` max | $100,000.00 / 272 / **0.0** | — | decoupled, DD col dead |

### Conservative ≡ Stress (byte-identical: equity 34,209,161 bytes each)

| Metric | Recalculated | summary.json |
| :--- | :--- | :--- |
| Total R / Σ USD | −13.7292934451 / −$10,253.59 | — |
| PF / Expectancy / Median R | 0.8815 / −0.0718811175 / −0.5159 | 0.8850167350442101 / −0.0718811175135023 |
| Ending equity (additive) | **$89,746.41 (−10.25%)** | $235,405.41 (+135.41%) | **CONTRADICTION** |
| `summary.json` `costs_used` | — | **{execution_mode: conservative_mode, cost_profile: stress}** (both folders) |

`TP01_ANNUAL_SUMMARY.csv` cross-check: ΣR −13.06, ΣUSD −$9,346.31, per-year DD 9.49/7.50/6.67/4.81% — aggregate maxDD cannot be 1.32% when single years draw down 4.8–9.5%.

---

## 6. PF / Expectancy / Equity Consistency

**Internally consistent (all losing):** Total R, expectancy, PF, win rate, median R, ΣUSD, per-year/month tables, `negative_years=3 / positive_years=1`, `negative_months=21–22`. **Mutually impossible:** `total_return_pct≈+135%`, `ending_equity≈$235k`, `max_dd≈1.3%`.

Root cause is **not** equity-level sign flip (PF/expectancy are consistently negative and track cost differences). It is an **equity/return/drawdown aggregation decoupled from the ledger** (equity rises while trades lose; min == start; `drawdown_pct` all zeros; same corruption in engine-native `summary.json`). A **second independent defect** is the **49% ledger sign inversion**. **Reconciliation: FAILED** → `TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`.

---

## 7. Annual Activity Audit

From `trades.csv` `entry_time_ny` (all profiles identical): **191** — 2015 **57** · 2016 **66** · 2017 **63** · 2018 **5** · 2019–2024 **0**. Corroborated by `TP01_ANNUAL_SUMMARY.csv`, `MONTHLY_SUMMARY` (ends 2018-01), `TRADE_DISTRIBUTION_SUMMARY` (2,931 zero-trade days). Genuine property of the output.

---

## 8. Post-2018 Inactivity Assessment

Dossier blames volatility-regime compression vs the ATR-50th-pct + `momentum ≥ 1.5×ATR` filter in 08:00–12:00 NY. Supporting checks (`check_loaded_frame.py`, "no NaN/inf 2018–2024", indicator population) are **asserted, not reproduced** in artifacts → **not accepted as evidence**. Distinguishing regime drift vs over-restrictive filter vs timezone/loader/cache/condition bug needs diagnostic re-execution / engine inspection — **forbidden in scope**.

Classification: **`BLOCKED_ACTIVITY_DIAGNOSIS_INSUFFICIENT`** (cause uncertifiable within scope). Practical gatekeeper read: **`STRATEGY_OBSOLETE_IN_RECENT_REGIME`** (0 trades in 6 of 10 years ⇒ not forward-deployable regardless of cause). Either way it does not unblock the dossier.

---

## 9. Cost Profile Audit

| Folder | summary.json `cost_profile` | `execution_mode` | Distinct? |
| :--- | :--- | :--- | :--- |
| base | base | normal_mode | YES |
| conservative | **stress** | **conservative_mode** | NO — dup of stress |
| stress | stress | conservative_mode | NO — same as conservative |

Base vs non-base ledgers differ (cost model partially functional). **conservative ≡ stress byte-identical** (trades, equity 34,209,161 B, PF, expectancy, every year) and **both** self-report `cost_profile=stress / conservative_mode`. No genuine conservative run; stress never executed in a stress mode. `RUN_MANIFEST.cost_profiles_run=[base,conservative,stress]` with 3 distinct rows is **misleading** (conservative & stress rows carry identical numbers). Classification: **`COST_PROFILE_APPLICATION_BLOCKED`** (base valid; conservative non-existent; stress mislabeled). True stress sensitivity is **unknown**.

---

## 10. Safety / Leakage Audit

Token scan over rerun report + run folder + manifest; hits classified:

| Hit | Location | Class |
| :--- | :--- | :--- |
| "zero holdout / zero 2025/2026 / no news / no high-precision" | report L13 | negative declaration |
| "sealed_holdout_2025_2026 never opened" | report L88 | negative declaration |
| "NewsConfig(enabled=False), news unreferenced" | report L89 | negative declaration |
| `holdout_used:false`, `optimization_run:false` | RUN_MANIFEST | negative declaration |
| "100% equivalence", "no NaN/inf 2018–2024" | report L39/L66 | unverified narrative (rejected as evidence) |
| `2026-05-1x`, branch, run_id | report/manifest | benign metadata |

**No blocker-class hits.** Independent confirmation: `trades.csv` 2015–2018 only; `equity_curve.csv` last datetime `2024-12-31 17:00:00`; data_range `2015-01-01..2024-12-31`.

| Control | Verdict |
| :--- | :--- |
| holdout_used | **false** (manifest + `no_holdout=true`) |
| 2025_2026_used | **false** (no rows beyond 2024-12-31) |
| validation_run | **false** | optimization_run | **false** (empty `optimization_results.csv`) |
| sweep_run | **false** | news_used | **false** (`news_filter_used:false`) |
| high_precision_used | **false** (`price_source='bid'`) |
| engine_modified (by audit) | **false** | data_modified (by audit) | **false** |
| new run by auditor | **false** |

Leakage posture: **PASS on data leakage**. Failure is **metric & cost-label integrity**, not leakage.

---

## 11. Final Classification

`TP01_BLOCKED_METRIC_INCONSISTENCY`

Equity decoupled + 49% sign inversion + self-contradictory engine `summary.json` + cost-profile mislabel/duplication. Inputs corrupted ⇒ no *final* `REJECTED_LOW_EDGE` / `REJECTED_REGIME_OBSOLESCENCE` verdict (those assume trustworthy metrics). BLOCKED supersedes. Not champion, not portfolio candidate, not validation/holdout-eligible.

---

## 12. Decision

- **Dossier REJECTED / BLOCKED.** `TP01_FORMAL_RERUN_SUCCESS_AND_SEALED` **overturned**; `+135.71%`, `1.32% maxDD`, "3 cost profiles" **rejected as corrupt artifacts**. "100% equivalence" certification **void** (equivalence to a corrupted baseline; unverifiable).
- **TP-01 cannot advance**: no champion, no rentability, no incubation, no FTMO/demo/real, no portfolio candidacy, no validation/holdout.
- **MR-01 formal cannot proceed yet** — defects are in the shared engine metric/cost layer; any new run inherits identical corruption.
- All prior harness dossiers are **suspect** until the metric/cost layer is fixed and reconciled against hand-computed ledgers.

---

## 13. Recommendation

- TP-01: **BLOCKED on metric integrity** + (defect-independent) **negative edge** (PF<1, expectancy −0.068R, median R −0.55, 3 neg/1 pos year) + **2019–2024 inactivity** ⇒ **remove from first wave**; **requires metric-engine remediation AND strategy/regime redesign**. Not watchlist-as-promising; not a portfolio candidate.
- Priority is the **shared metric/cost-layer fix**, not TP-01. Until fixed: **no dossier may be sealed** and **MR-01 formal is gated**.
- Add a mandatory **ledger↔dossier reconciliation gate** (independent recompute within strict tolerance; sign invariants `stop_loss⇒loss`, `take_profit⇒win`; equity must derive from the ledger; each cost profile must self-report its own profile/mode and be a genuinely distinct run) before any future `*_SUCCESS_AND_SEALED`.

---

## 14. Copy-Paste Summary for ChatGPT

```
TP-01 formal train-only dossier (2015–2024) EXTERNAL AUDIT v3 (3rd deterministic reproduction, commit 66e42063) → STATUS: TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY.
Base branch research/tp01-formal-train-only-rerun-after-performance-fix-20260516; audit branch audit/tp01-formal-dossier-external-audit-v3-20260517 (v1+v2 already on origin; no force-push).
4 shared-engine-layer defects: (1) equity_curve decoupled from ledger — ends $235,710.51 (+135.71%), drawdown_pct column all zeros, min never below $100k, while trades.csv sums -13.06R / -$9,346.31; (2) sign inversion 94/191 (49.2%) — 63 stop_loss labeled win, 17 take_profit labeled loss; (3) engine summary.json self-contradictory (PF<1, exp<0, 3 neg/1 pos year, 21–22 neg months, yet +135%); (4) cost-profile mislabel — base=base/normal_mode but BOTH conservative/ and stress/ summary.json = stress/conservative_mode, equity files byte-identical → only 2 real runs, no genuine conservative, stress never run in stress mode (manifest "3 profiles" false).
Recalc base: 191 trades, win rate 47.64%, PF 0.889 (<1), expectancy -0.0684R, total R -13.06, median R -0.55, true ending equity ≈ $90,654 (-9.35% additive) / ≈$93,326 (-6.67% compounded), real maxDD ≈ 8.5%. Activity: 2015:57, 2016:66, 2017:63, 2018:5, 2019–2024:0.
Safety: no data leakage (no holdout/2025-26/news/opt/sweep/high-precision; trades 2015–2018 only); equivalence & data-completeness claims rejected as unverified narrative. Git/output policy COMPLIANT (docs/light only; heavy CSV git-ignored). Results reproduced identically across 3 independent audit passes → systematic, not noise.
Decision: dossier overturned; TP-01 removed from first wave; NOT champion/portfolio/incubation. MR-01 formal GATED behind a shared metric/cost-layer fix + a mandatory ledger↔dossier reconciliation gate.
Next: NEXT_PROMPT_FIX_TP01_METRIC_INCONSISTENCY.md (primary); NEXT_PROMPT_FORMAL_TRAIN_ONLY_MR01_BACKTEST.md (blocked until fix lands).
```

---
*External audit v3 complete. Dossier blocked on metric & cost-label integrity; findings deterministically reproduced. No new runs, optimizations, validations, or data/engine changes were performed.*
