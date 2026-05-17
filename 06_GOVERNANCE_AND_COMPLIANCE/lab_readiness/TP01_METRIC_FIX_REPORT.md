# TP01 METRIC FIX REPORT

**ROLE**: Metric Integrity Remediation Engineer · PnL accounting · backtest correctness auditor · Python test engineer · cost-model auditor · repo safety officer · quant gatekeeper
**DATE**: 2026-05-17
**BASE BRANCH**: `audit/tp01-formal-dossier-external-audit-v3-20260517`
**FIX BRANCH**: `fix/shared-metric-cost-integrity-20260517`
**AUDITED COMMIT (input)**: `767c282d9ef4d98dddd7e42df158b2343f4dabcb`
**TRIGGER**: `TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY` (external audit v3)

---

## 1. Status

`TP01_METRIC_FIX_PARTIAL_OWNER_REVIEW_REQUIRED`

The two core shared-layer integrity defects (PnL directional sign; equity/drawdown/summary decoupling) are **fixed in `engine.py`, unit-tested, and covered by a new reconciliation gate**, with **no regression** in the existing engine suites. The cost-profile defect is **precisely diagnosed but deliberately NOT auto-fixed**: a genuine third cost tier requires an owner cost-model decision (new engine enums + cost parameters) and a reviewed change to a scratch harness — outside safe autonomous scope. No backtest/strategy/optimization/sweep/validation was run; the corrupt TP-01 artifact is left intact and is now *correctly flagged as suspect* by the gate.

---

## 2. Executive Summary

Black-box audit symptoms were traced to **three concrete code defects, all in `03_RESEARCH_LAB/research_lab/engine.py`** (the audit's "summary.json self-contradiction" is a *consequence* of the equity defect, not a separate bug; `report.py` math was already correct and ledger-derived):

1. **PnL directional sign.** `units` is computed unsigned (`risk_usd / (abs(stop_distance) * q2usd)`, `engine.py:~857`) and the realized-PnL formula had no direction factor, so **every short** (`Short Trades Count = 94`, exactly the audit's 94/191 "sign-inverted") had inverted `pnl_usd`/`pnl_r`; losing shorts (hit `stop_loss`, price rose) showed `+pnl` ⇒ `result=win`, winning shorts (`take_profit`, price fell) showed `−pnl` ⇒ `result=loss`.
2. **Equity inflation.** At entry only commission leaves `cash` (`engine.py:~861`); `risk_usd` is **never reserved**. At exit the code did `cash += (position.risk_usd + pnl_usd)`, adding back a reservation that never happened → `cash` compounded ≈ +0.5%/trade regardless of PnL. ≈`(1.005)^191 ≈ 2.6` explains the fake `+135.71%` / `$235,710.51`.
3. **Equity curve / drawdown.** `equity_points` recorded raw inflated `cash` with explicitly zero unrealized PnL (`engine.py:~1071`, `# Simplificado`) → near-monotone rising series, `drawdown_pct` ≈ all-zero. `report.py` then faithfully derived a fake `+135%`/`1.3% maxDD` from that corrupt series, while PF/expectancy (computed from the also-corrupt-but-independently `pnl` ledger) stayed negative → the "summary.json self-contradiction".

Fixes 1 & 2 are mathematically unambiguous, localized, and fully testable without any backtest; fix 3 is resolved automatically once `cash` is correct (the per-bar `equity = cash` becomes true realized equity and `report.py` then produces real drawdowns).

The **cost-profile defect is harness-level, not engine-level**: `03_RESEARCH_LAB/scratch/formal_run_tp01.py:~179-209` hardcodes both the `conservative` **and** `stress` profiles to identical `execution_mode="conservative_mode", cost_profile="stress"`. Moreover `config.py:45-44` exposes **no `conservative` cost tier** (`SUPPORTED_COST_PROFILES = auto, base, stress, precision`) and **no `stress_mode`** (`SUPPORTED_EXECUTION_MODES = normal_mode, conservative_mode, high_precision_mode`). A genuine 3-way differentiation therefore needs an **owner decision on cost-model design** (new enum members + spread/slippage parameters) — inventing those numbers is a quant risk decision, not a mechanical fix → left for owner review.

---

## 3. Defects Fixed

| # | Defect | File / site | Status |
| :- | :- | :- | :- |
| D1 | Short PnL sign inverted (unsigned `units`, no direction factor) | `engine.py` exit (normal + final-bar) | **FIXED** |
| D2 | Equity inflated by spurious `+ risk_usd` add-back never reserved at entry | `engine.py` exit (normal + final-bar) | **FIXED** |
| D3 | `result` label / `pnl_r` sign / `stop_loss⇒win` / `take_profit⇒loss` | consequence of D1 (`report.py:85` logic already correct) | **FIXED via D1** |
| D4 | equity_curve decoupled / `drawdown_pct` dead / fake `total_return` | consequence of D2 (`report.py` math already correct) | **FIXED via D2** |
| D5 | summary.json self-contradiction (PF<1 & exp<0 yet +135%) | consequence of D2/D4 | **FIXED via D2** |
| D6 | Cost profiles mislabeled/duplicated (conservative ≡ stress) | `scratch/formal_run_tp01.py:~179-209` + missing engine cost tiers | **DIAGNOSED — OWNER REVIEW** |

---

## 4. Files Changed

| File | Change |
| :- | :- |
| `03_RESEARCH_LAB/research_lab/engine.py` | +18 / −4. New pure helper `directional_pnl_usd(direction, entry, exit, units, q2usd)`; both exit paths route through it; `cash += (position.risk_usd + pnl_usd)` → `cash += pnl_usd` (both paths) |
| `03_RESEARCH_LAB/research_lab/metric_reconciliation.py` | **NEW** — pure ledger↔dossier reconciliation gate (no IO, no market data) |
| `03_RESEARCH_LAB/research_lab/tests/test_metric_reconciliation.py` | **NEW** — 19 tests (invariants 1–13) |

No engine logic outside the PnL/cash sites was touched. Signal logic, strategy code, MR-01, data vault: untouched.

---

## 5. PnL Sign Invariants (enforced + tested)

- `long`: profit ⇔ `exit_price > entry_price`. `short`: profit ⇔ `exit_price < entry_price` (via `direction_sign = +1 long / −1 short`; `units` stays positive so `lots`/commission are unaffected).
- `stop_loss ⇒ pnl ≤ 0 ⇒ result=loss`; `take_profit ⇒ pnl ≥ 0 ⇒ result=win`.
- `result` derives from net `pnl_usd` sign (not from `exit_reason`); `pnl_r` shares the sign of `pnl_usd` (`pnl_r = pnl_usd/risk_usd`, `risk_usd>0`).
- costs only worsen PnL (`pnl_usd -= exit_commission_usd`; entry commission already removed from cash).
- Tests 1–6 + `reconcile_trades` (`PNL_SIGN_MISMATCH`, `STOP_LOSS_POSITIVE_PNL`, `TAKE_PROFIT_NEGATIVE_PNL`, `RESULT_LABEL_MISMATCH`).

## 6. Equity / Drawdown Reconciliation

- `cash` now changes only by realized `pnl_usd` (and entry commission at entry). Per-bar `equity = cash` ⇒ true realized equity; `report.py` `drawdown_pct = (equity/cummax−1)*100` now reflects real losses.
- Invariant: `ending_equity ≈ starting_equity + Σ pnl_usd` (additive; no hidden compounding). `max_dd ≥` worst sub-period DD. A net-losing ledger **must** show non-zero drawdown.
- Gate: `reconcile_equity` → `ENDING_EQUITY_DECOUPLED`, `DRAWDOWN_DEAD`, `MAX_DRAWDOWN_MISMATCH`, `TOTAL_RETURN_SIGN_DECOUPLED`. Tests 7, 8, 8b, total-return-sign.
- Expected post-fix TP-01 (from offline recompute, indicative only — NOT re-run): ≈ **−$9.3k / ≈ −9% additive**, real maxDD ≈ 8.5%, not +135%.

## 7. summary.json Reconciliation

`reconcile_summary` fails when `PF<1 & expectancy<0 & total_return>0` (`SUMMARY_SELF_CONTRADICTION`). Test 9/9b. Test 12 feeds the **committed** `TP01_COST_PROFILE_SUMMARY.csv` and asserts the pre-fix artifact is flagged ⇒ the existing TP-01 dossier remains **suspect** until regenerated under this fix.

## 8. Cost Profile Routing

**Not auto-fixed (owner review).** Diagnosis: `scratch/formal_run_tp01.py` sets `conservative` and `stress` to byte-identical config; engine lacks a `conservative` cost tier and a `stress_mode`. The gate (`reconcile_cost_profiles`) **does** detect the mislabel/duplicate (`COST_PROFILE_MISLABEL`, `COST_PROFILE_DUPLICATE`; tests 10/11). Required owner decisions before any 3-profile re-run: (a) add a real `conservative` cost tier + (optional) `stress_mode` to `config.py`/engine cost model with explicit spread/slippage multipliers; (b) correct the harness mapping; (c) make each `summary.json` self-report its own profile; (d) `RUN_MANIFEST` must list only genuinely-distinct profiles run.

## 9. Reconciliation Gate

`research_lab/metric_reconciliation.py` — pure functions (`reconcile_trades`, `reconcile_equity`, `reconcile_summary`, `reconcile_cost_profiles`, `reconcile_all`). No imports of pandas/numpy/IO; runs on synthetic fixtures or already-produced artifacts; **no backtest/data/holdout/2025-26 needed** (AST-enforced by test 13). Intended as a mandatory pre-seal gate: no `*_SUCCESS_AND_SEALED` may be issued while any violation is non-empty.

## 10. Tests

`$env:PYTHONPATH="03_RESEARCH_LAB"`

| Suite | Result |
| :- | :- |
| `test_metric_reconciliation.py` (new, 19) | **19/19 OK** |
| `test_engine.py` (regression) | **17/17 OK** |
| `test_engine_stop_entry.py` (regression) | **3/3 OK** |
| `test_lab_preflight_no_leakage.py` (regression) | **6/6 OK** |
| **Total** | **45 pass, 0 fail** |

Invariants 1–13 covered. Regression suites confirm the `engine.py` edits do not alter existing engine behavior/contracts.

## 11. Safety Verification

- backtest_run: NO
- strategy_run: NO
- optimization_run: NO
- sweep_run: NO
- validation_run: NO
- holdout_used: NO
- 2025_2026_used: NO
- news_used: NO
- high_precision_used: NO
- data_modified: NO
- force_push: NO
- git_add_dot_used: NO

Static scan hits in changed/new files = docstring/comment **negative declarations** + one synthetic `high_precision_mode` label and the light committed-CSV path in tests → **all benign; no blocker**. Heavy outputs / `local_outputs_do_not_commit` / ZIP / data: not touched, not staged. Unrelated dirty files (`strategy_research_intake/`): not staged.

## 12. Remaining Risks

- **No end-to-end re-run validation** (forbidden in scope): fixes are proven by unit tests + offline recompute, not by a fresh formal backtest. TP-01's existing artifact stays corrupt (correctly flagged suspect).
- **Cost-profile genuine differentiation unresolved** — owner cost-model decision required; until then any "3 cost profiles" run is invalid.
- Final-bar-close path updates `cash` after the loop, so the very last trade isn't appended as an extra equity point (pre-existing minor granularity; immaterial for TP-01 whose last trade is 2018-01 with no open position at series end). Documented, not expanded.
- Other strategies/dossiers produced pre-fix remain **suspect** until regenerated under this fix and passed through the gate.

## 13. Next Step

Core metric fix is effectively done ⇒ proceed to artifact reconciliation/regeneration (gated). Created:
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_RECONCILE_TP01_ARTIFACTS_AFTER_METRIC_FIX.md`
— which additionally hard-gates on the **cost-profile owner decision** before any 3-profile formal re-run, and keeps MR-01 blocked until TP-01 is regenerated clean and gate-green.

---
*Metric fix partial-complete. PnL sign + equity/drawdown/summary repaired and gated; cost-profile differentiation escalated to owner review. No backtest, optimization, validation, data or engine-design changes beyond the two localized PnL/cash corrections were performed.*
