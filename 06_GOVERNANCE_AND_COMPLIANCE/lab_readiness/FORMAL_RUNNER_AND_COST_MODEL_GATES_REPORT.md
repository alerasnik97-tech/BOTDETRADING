# FORMAL RUNNER AND COST MODEL GATES REPORT

**ROLE**: Institutional Research Infrastructure Architect · Formal Backtest Runner Engineer · Metric Reconciliation Gatekeeper · Cost Model Governance Officer · Data Leakage Prevention Auditor · Python Test Engineer · Git Safety Officer · Quant Lab Release Gatekeeper
**DATE**: 2026-05-17
**BASE BRANCH**: `fix/institutional-cost-profile-routing-20260517`
**INFRA BRANCH**: `infra/formal-runner-cost-gates-20260517`
**INPUT COMMIT**: `4b9e799ef6c4c362b64c69015728f957f58dd67d`

---

## 1. Status

`FORMAL_RUNNER_AND_COST_GATES_READY_FOR_TP01_REGENERATION`

GATE 1 (conservative cost model owner-ratified, research-only) and GATE 2 (versioned, committable, fail-closed formal runner outside `scratch/`) are both closed. 97/97 tests green, static scan clean, no execution performed.

---

## 2. Executive Summary

Both blocking gates are resolved as **infrastructure + governance only**. No backtest, strategy run, optimization, sweep, validation, holdout, news, high-precision or data access occurred.

- **GATE 1** — `COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md` records the owner ratification: `conservative` ×1.20 spread / ×1.30 slippage **research/train-only** (explicitly NOT real/FTMO/demo/production/incubation/deployment); `stress` keeps pre-existing institutional ×1.35/×1.60; commissions unchanged; `base < conservative < stress`.
- **GATE 2** — `research_lab/runners/formal_train_runner.py` (+ `runners/__init__.py`) is the new single sanctioned, sealable mechanism. Import-side-effect-free, dry-run/fail-closed by default; rejects holdout/sealed_holdout/2025-26/validation/optimization/sweep/news/high-precision/`precision` profile/non-train-only and non-`prepared_train_2015_2024` data paths; enforces the output area (no root/data-vault/production/incubation/scratch/ZIP); refuses to seal unless the reconciliation gate returns zero violations and a manifest exists.

---

## 3. Owner Cost Decision

`COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md` → `COST_MODEL_OWNER_DECISION_RECORDED_RESEARCH_ONLY`. conservative 1.20/1.30 research-only; stress 1.35/1.60 unchanged (pre-existing, not invented); commissions 7.0 USD unchanged; strict `base<conservative<stress`; NOT approved for real/FTMO/demo/production/incubation/deployment; revisable only via a future broker/FTMO-evidenced phase.

## 4. Official Runner Created

`03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py` (+ `__init__.py`). Surface: `FormalRunRequest`, `build_cost_profile_configs`, `validate_cost_profile_configs`, `validate_train_only_scope`, `assert_safe_request`, `validate_output_dir`, `heavy_output_dir`, `build_run_manifest`, `reconcile_profile_outputs`, `seal_run_only_if_reconciled`, `preflight`, `run_single_strategy_formal_train_only`, CLI `build_arg_parser`/`main`. **Replaces** the git-ignored `scratch/formal_run_tp01.py` as the only sealable runner.

## 5. Runner Safety Contract

| Property | Guarantee | Test |
| :- | :- | :- |
| Import side-effects | none; engine/data/strategies/report lazy-imported in execute path only | 01, 01b, 30 |
| Default mode | `execute=False` → dry-run preflight, aborts before backtest | 02 |
| `__main__` guard | CLI runs only as a script, never on import | 01b |
| Holdout / sealed_holdout / validation in path | rejected by `validate_train_only_scope` | 11, 12, 13 |
| Non-`prepared_train_2015_2024` data | rejected | 13b |
| 2025/2026 dates | rejected (`>= 2025-01-01`) | 14, 29 |
| optimization / sweep / validation / holdout / news / high-precision / non-train-only flags | rejected fail-closed | 15, 16, 17, 17b |
| Output policy | under `…/reports/formal_train_only/…`; no root/data-vault/production/incubation/scratch/ZIP | 18, 19, 20, 20b, 21, 22 |
| Heavy artifacts | routed under `local_outputs_do_not_commit/<profile>` | 23 |

## 6. Cost Profile Contract

`build_cost_profile_configs` → exactly base/normal_mode, conservative/conservative_mode, stress/stress_mode (no `precision`). `validate_cost_profile_configs` enforces self-report, uniqueness (no duplicates), no high-precision mode, no `precision`/`auto` cost profile, and strict monotonicity `1.0 < conservative_mult < stress_mult` (spread + slippage). Tests 03–10.

## 7. Reconciliation Gate Contract

`reconcile_profile_outputs` wraps `metric_reconciliation.reconcile_all`; `seal_run_only_if_reconciled` raises `ReconciliationGateError` when any violation exists, when no reconciliation was performed, or when the manifest is missing. Clean synthetic ledger seals; contradictory/empty does not. Tests 24, 24b, 24c, 25. The execute path reconciles every profile before sealing.

## 8. Tests

`$env:PYTHONPATH="03_RESEARCH_LAB"`

| Suite | Result |
| :- | :- |
| `test_formal_train_runner_contract.py` (new) | **41/41 OK** |
| `test_cost_profiles.py` | 11/11 OK |
| `test_metric_reconciliation.py` | 19/19 OK |
| `test_engine.py` | 17/17 OK |
| `test_engine_stop_entry.py` | 3/3 OK |
| `test_lab_preflight_no_leakage.py` | 6/6 OK |
| **Total** | **97 pass, 0 fail** |

41 = the 30 numbered contracts + complementary positive/negative cases. Manifest records only profiles run and rejects duplicates/empty (26, 27, 27b). No-data / no-2025-26 / no-strategy-run guarantees verified structurally + by subprocess (28, 29, 30).

## 9. Static Scan

Tokens scanned across `runners/`, new tests, `lab_readiness/`. Every hit classified **intended guard** (date/holdout/zip/scratch/production/incubation rejection logic, banned-mode/profile constants), **negative declaration** (docstrings of what is forbidden; owner-doc "Safety: NO"), or **benign** (dates, branch refs, test fixtures feeding rejected strings). The only data-loader reference is `load_backtest_data_bundle` inside the **lazy, gated, execute-only** branch — unreachable on import or in dry-run/tests (proven by subprocess tests 01/30). **No blocker.**

## 10. Files Changed

| File | Type |
| :- | :- |
| `03_RESEARCH_LAB/research_lab/runners/__init__.py` | new |
| `03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py` | new |
| `03_RESEARCH_LAB/research_lab/tests/test_formal_train_runner_contract.py` | new |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md` | new |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_AND_COST_MODEL_GATES_REPORT.md` | new |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER.md` | new |

No engine/config/strategy/data change this phase. `scratch/` untouched/uncommitted. Unrelated `strategy_research_intake/` dirty files not staged.

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

## 12. Decision

- **TP-01 regeneration is AUTHORIZED** for the next phase, **using only the official runner** under the mandatory reconciliation gate.
- **MR-01 remains BLOCKED** until TP-01 is regenerated clean and gate-green (and externally audited).
- The official runner **replaces** `scratch/formal_run_tp01.py` as the sole sanctioned, committable, sealable mechanism.
- `conservative` multipliers are **owner-approved research/train-only** (NOT real/FTMO/demo/production).
- Nothing missing for the gate; one honest caveat below.

## 13. Honest Caveat

The runner's `execute=True` path is correct-by-construction and lazy-wires the real engine/data/strategies, but per phase rules it was **not exercised end-to-end** (no backtest). Full execution validation occurs in the next phase, where the reconciliation gate is the hard backstop. TP-01 remains, defect-independent, a rejection candidate (PF<1, expectancy<0, 0 trades 2019–2024); this phase only restores infrastructure/governance integrity.

## 14. Next Step

`READY` ⇒ created `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER.md`.

---
*Both gates closed as infrastructure + governance. No backtest, strategy run, optimization, sweep, validation, holdout, news, high-precision or data change was performed.*
