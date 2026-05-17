# COST PROFILE OWNER DECISION AND ROUTING FIX REPORT

**ROLE**: Institutional Cost Model Architect · FX execution-cost auditor · FTMO risk specialist · metric-integrity engineer · Python test engineer · repo safety officer
**DATE**: 2026-05-17
**BASE BRANCH**: `fix/shared-metric-cost-integrity-20260517`
**FIX BRANCH**: `fix/institutional-cost-profile-routing-20260517`
**INPUT COMMIT**: `69cf14af959344788d039ca1f4ca23cd01d7d92b`
**TRIGGER**: `TP01_METRIC_FIX_PARTIAL_OWNER_REVIEW_REQUIRED` (cost-profile owner decision pending)

---

## 1. Status

`COST_PROFILE_PARTIAL_OWNER_REVIEW_REQUIRED`

The structural defect (no real `conservative` tier, no `stress_mode`, `conservative_mode` auto-mapped to `"stress"`, conservative≡stress duplication) is **fixed in the committable shared layer (`config.py` + `engine.py`), routed, and tested** (56/56 green, no regression). Two items remain for the owner: (a) **ratify the provisional `conservative` multiplier magnitudes** (`OWNER_APPROVED_DEFAULTS_REQUIRED`); (b) **provide an official, committable formal runner** — the existing harness `03_RESEARCH_LAB/scratch/formal_run_tp01.py` is **git-ignored by design** (`.gitignore:41 scratch/`), so its routing correction was applied and validated locally but **cannot be the sealed mechanism**.

---

## 2. Executive Summary

Cost differentiation is driven solely by `resolved_cost_profile(engine_config)` consumed in `engine.estimate_spread_pips` / `estimate_slippage_pips`. Before this fix: `"conservative"` was **not a real branch** (behaved exactly like `base`), there was **no `stress_mode`**, and `resolved_cost_profile` / `with_execution_mode` auto-mapped `conservative_mode → "stress"` — the root of the audit's conservative≡stress duplication. `"stress"` already had **institutional parameters** (`stress_spread_multiplier=1.35`, `stress_slippage_multiplier=1.6`); these were **kept, not invented**. A real `conservative` tier was added with documented defaults strictly between `base` and the existing `stress` ceiling. Three genuinely distinct, monotone, self-reporting profiles now exist and are enforced by the reconciliation gate.

No backtest, strategy run, optimization, sweep, validation, holdout, news, high-precision or data change was performed.

---

## 3. Cost Model Decision

| Parameter | Value | Provenance |
| :- | :- | :- |
| base spread / slippage | `DEFAULT_SPREAD_PIPS=1.2` / `DEFAULT_SLIPPAGE_PIPS=0.2`, no profile multiplier | existing default — unchanged |
| `stress_spread_multiplier` | **1.35** | **EXISTING institutional param — not invented** |
| `stress_slippage_multiplier` | **1.60** | **EXISTING institutional param — not invented** |
| `conservative_spread_multiplier` | **1.20** | NEW — `OWNER_APPROVED_DEFAULTS_REQUIRED` |
| `conservative_slippage_multiplier` | **1.30** | NEW — `OWNER_APPROVED_DEFAULTS_REQUIRED` |
| commissions | `DEFAULT_COMMISSION_ROUNDTURN_USD=7.0`, all profiles | unchanged (per "no tocar comisiones") |

**Why conservative = 1.20 / 1.30 (not the prompt's generic "1.25–1.50"):** the binding institutional constraints are (i) strict monotonicity `base < conservative < stress` and (ii) the *existing* stress ceiling (spread ×1.35, slippage ×1.60). A 1.50 spread multiplier would **exceed** the institutional stress spread (1.35) and break ordering. 1.20 / 1.30 sits cleanly between `base` (×1.00) and `stress` (×1.35 / ×1.60): harder than base, realistic for funded/retail, below stress. These two magnitudes are **provisional defaults requiring owner ratification** before any real-capital or FTMO decision; they do not block train-only structural correctness.

Verified routing on a neutral mid-session bar (no late/high-vol multipliers): spread `1.2000 / 1.4400 / 1.6200`, slippage `0.2000 / 0.2600 / 0.3200` for base / conservative / stress — strictly monotone, never reducing cost.

---

## 4. Profiles Defined

| Profile | `cost_profile` | `execution_mode` | spread×/slip× | distinct |
| :- | :- | :- | :- | :- |
| BASE | `base` | `normal_mode` | 1.00 / 1.00 | yes |
| CONSERVATIVE | `conservative` | `conservative_mode` | 1.20 / 1.30 | yes |
| STRESS | `stress` | `stress_mode` | 1.35 / 1.60 | yes |

Each self-reports its own profile via `resolved_cost_profile`; `conservative_mode → "conservative"` (was `"stress"`), `stress_mode → "stress"` (new); `intrabar_policy` auto = `conservative` for both `conservative_mode` and `stress_mode` (pessimistic). No profile is a duplicate of another.

---

## 5. Files Changed

| File | Tracked? | Change |
| :- | :- | :- |
| `03_RESEARCH_LAB/research_lab/config.py` | yes | `SUPPORTED_EXECUTION_MODES` += `stress_mode`; `SUPPORTED_COST_PROFILES` += `conservative`; new `conservative_spread_multiplier=1.20` / `conservative_slippage_multiplier=1.30` (OWNER_APPROVED_DEFAULTS_REQUIRED); `resolved_cost_profile` & `with_execution_mode` fixed (`conservative_mode→conservative`, add `stress_mode→stress`, intrabar pessimistic for stress_mode) |
| `03_RESEARCH_LAB/research_lab/engine.py` | yes | `estimate_spread_pips` / `estimate_slippage_pips`: new `profile == "conservative"` branch applying the conservative multipliers |
| `03_RESEARCH_LAB/research_lab/tests/test_cost_profiles.py` | yes (new) | 11 routing/monotonicity/gate tests |
| `03_RESEARCH_LAB/scratch/formal_run_tp01.py` | **NO — git-ignored** | conservative→`cost_profile="conservative"`; stress→`execution_mode="stress_mode"`. Applied & validated **locally only**; NOT staged (scratch/ excluded by `.gitignore:41`). Owner must mirror this in an official runner. |

Signal logic, strategy code, MR-01, data vault, commissions: untouched.

---

## 6. Tests

`$env:PYTHONPATH="03_RESEARCH_LAB"`

| Suite | Result |
| :- | :- |
| `test_cost_profiles.py` (new, 11) | **11/11 OK** |
| `test_metric_reconciliation.py` (19) | **19/19 OK** |
| `test_engine.py` | **17/17 OK** |
| `test_engine_stop_entry.py` | **3/3 OK** |
| `test_lab_preflight_no_leakage.py` | **6/6 OK** |
| **Total** | **56 pass, 0 fail** |

Covers: base/conservative/stress self-report; `conservative_mode` no longer maps to `stress` (regression); monotonic non-reducing costs; conservative≠stress; gate detects mislabel & duplicate; fixed routing passes the gate. Engine/preflight regressions confirm config/engine edits are behavior-safe for existing contracts.

## 7. Reconciliation Gate Impact

`metric_reconciliation.reconcile_cost_profiles` continues to enforce `COST_PROFILE_MISLABEL` (folder not self-reporting its own profile) and `COST_PROFILE_DUPLICATE` (identical config across profiles). Post-fix mapping passes the gate cleanly; the pre-fix duplicated/mislabeled config still fails it. Any future `RUN_MANIFEST` with duplicate profiles is rejected by this gate before sealing.

## 8. Safety Verification

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

Static scan of the diff/new files: only one hit — `high_precision_mode` retained inside the `SUPPORTED_EXECUTION_MODES` tuple (pre-existing enum member, not introduced/invoked) → **benign**; plus a test-file docstring **negative declaration**. No blocker. `scratch/` (git-ignored), `local_outputs_do_not_commit/`, heavy CSV, ZIP, root files, unrelated `strategy_research_intake/` dirty files: not staged.

## 9. Remaining Risks

- **`conservative` multiplier magnitude is provisional** (`OWNER_APPROVED_DEFAULTS_REQUIRED`). Structurally correct and monotone, but the exact 1.20/1.30 must be ratified by the owner before any funded/real interpretation.
- **Official runner gap**: the only formal runner is `scratch/formal_run_tp01.py`, which is git-ignored by design. A committable, reviewed runner that consumes the corrected routing is required before any sealed 3-profile regeneration. The local scratch edit is documented above so the owner can mirror it.
- No end-to-end re-run validation (forbidden in scope): correctness shown by unit tests + deterministic routing checks, not a fresh backtest.
- TP-01 remains, defect-independent, a rejection candidate (PF<1, expectancy<0, 0 trades 2019–2024); this work only restores cost-model integrity, it does not rehabilitate the strategy. MR-01 stays blocked.

## 10. Next Step

Structural routing is fixed and gated ⇒ proceed toward regeneration, hard-gated on the two owner items. Created:
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_REGENERATE_TP01_AFTER_METRIC_AND_COST_FIX.md`
— gated on (a) owner ratification of conservative multipliers and (b) an official committable runner; keeps MR-01 blocked until TP-01 is regenerated clean and gate-green.

---
*Cost-profile routing repaired in the committable shared layer and tested; conservative magnitude + official runner escalated to owner. No backtest, optimization, validation, news, high-precision, data or commission changes were performed.*
