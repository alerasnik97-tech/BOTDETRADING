# FORMAL RUNNER AND COST MODEL GATES REPORT

**ROLE**: Institutional Research Infrastructure Engineer · Formal Runner Architect · Cost Model Governance Officer · Metric Reconciliation Gatekeeper · Python Test Engineer · Git Safety Officer · Quant Lab Auditor
**DATE**: 2026-05-17
**BASE BRANCH**: `fix/institutional-cost-profile-routing-20260517`
**INFRA BRANCH**: `infra/formal-runner-and-cost-model-gates-20260517`
**INPUT COMMIT**: `4b9e799ef6c4c362b64c69015728f957f58dd67d`

---

## 1. Status

`FORMAL_RUNNER_AND_COST_GATES_READY_FOR_TP01_REGENERATION`

Both blocking gates are closed: (GATE 1) the `conservative` cost model is owner-ratified for research/train-only; (GATE 2) a versioned, committable, fail-closed formal runner now exists outside `scratch/`. 83/83 tests green, static scan clean, no execution performed.

---

## 2. Executive Summary

The two items that previously blocked any formal re-run are resolved as **infrastructure + governance only** — no backtest, strategy run, optimization, sweep, validation, holdout, news, high-precision or data access occurred.

- **GATE 1** — `COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md` records the owner decision: `conservative` ×1.20 spread / ×1.30 slippage ratified **for research/train-only only** (explicitly NOT real/FTMO/demo/production); `stress` keeps pre-existing institutional ×1.35/×1.60; commissions unchanged; `base < conservative < stress` enforced.
- **GATE 2** — `research_lab/runners/formal_train_runner.py` (+ `runners/__init__.py`) is the new single sanctioned mechanism. It is import-side-effect-free, dry-run/fail-closed by default, rejects holdout/2025-26/validation/optimization/sweep/news/high-precision, enforces the output policy, and refuses to seal unless the reconciliation gate returns zero violations. It orchestrates the already-fixed engine/cost/reconciliation layers and duplicates no strategy/engine logic.

---

## 3. Owner Cost Decision

Recorded in `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md` (`COST_MODEL_OWNER_DECISION_RECORDED_RESEARCH_ONLY`). D1 conservative 1.20/1.30 research-only; D2 stress 1.35/1.60 unchanged; D3 commission 7.0 USD unchanged; D4 `base<conservative<stress`; D5 NOT real/FTMO/demo/prod/incubation/deploy; D6 future change needs a separate broker/FTMO-evidenced phase.

## 4. Official Runner Created

`03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py` (+ `__init__.py`). Public surface: `FormalRunRequest`, `build_cost_profile_configs`, `validate_cost_profile_configs`, `assert_safe_request`, `validate_output_policy`, `heavy_output_dir`, `build_run_manifest`, `reconcile_profile_outputs`, `seal_run_only_if_reconciled`, `preflight`, `run_single_strategy_formal_train_only`, CLI `build_arg_parser`/`main`. **Replaces** `scratch/formal_run_tp01.py` (git-ignored, cannot be sealed) as the only sanctioned runner.

## 5. Runner Safety Contract

| Property | Guarantee | Test |
| :- | :- | :- |
| Import side-effects | none; engine/data/strategies/report are lazy-imported inside the execute path only | t1 (subprocess), t1b (AST) |
| Default mode | `execute=False` → dry-run preflight, aborts before any backtest | t16, t16b |
| `__main__` guard | CLI runs only as a script, never on import | t1b |
| Holdout / sealed_holdout | rejected (request + data_path) | t11 |
| 2025/2026 dates | rejected (`>= 2025-01-01`) | t12 |
| validation / optimization / sweep / news / high-precision / non-train-only | rejected fail-closed | t13, t14, t15, t15b |
| Output policy | must live under `…/BOT_V2_DAYTIME_LAB/reports/…`; no ZIP, no data vault, no scratch | t9, t9b, t9c, t10 |
| Heavy artifacts | routed under `local_outputs_do_not_commit/<profile>` | t19 |

## 6. Cost Profile Contract

`build_cost_profile_configs` yields exactly base/normal_mode, conservative/conservative_mode, stress/stress_mode; `validate_cost_profile_configs` enforces self-report (`resolved_cost_profile == name`), uniqueness (no duplicates), no high-precision, and strict monotonicity `1.0 < conservative_mult < stress_mult` for both spread and slippage. Tests t2–t8.

## 7. Reconciliation Gate Contract

`reconcile_profile_outputs` wraps `metric_reconciliation.reconcile_all`; `seal_run_only_if_reconciled` raises `ReconciliationGateError` on any violation. A clean synthetic ledger seals; a contradictory one is blocked. Tests t17, t18. The execute path calls the gate per profile before sealing.

## 8. Tests

`$env:PYTHONPATH="03_RESEARCH_LAB"`

| Suite | Result |
| :- | :- |
| `test_formal_train_runner_contract.py` (new) | **27/27 OK** |
| `test_cost_profiles.py` | 11/11 OK |
| `test_metric_reconciliation.py` | 19/19 OK |
| `test_engine.py` | 17/17 OK |
| `test_engine_stop_entry.py` | 3/3 OK |
| `test_lab_preflight_no_leakage.py` | 6/6 OK |
| **Total** | **83 pass, 0 fail** |

(27 = the 20 numbered contracts + complementary positive/negative cases.)

## 9. Static Scan

Tokens scanned across `runners/`, new tests and `lab_readiness/`. Every hit classified: **intended guard** (date/holdout/zip/scratch/heavy-dir rejection logic, banned-mode constants), **negative declaration** (docstrings stating what is forbidden), or **benign** (dates, branch refs). **No blocker** — no real holdout read, no 2025/2026 data, no news/high-precision/optimization/sweep execution, no ZIP write, no scratch commit, no `git add .`.

## 10. Files Changed

| File | Type |
| :- | :- |
| `03_RESEARCH_LAB/research_lab/runners/__init__.py` | new |
| `03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py` | new |
| `03_RESEARCH_LAB/research_lab/tests/test_formal_train_runner_contract.py` | new |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md` | new |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_AND_COST_MODEL_GATES_REPORT.md` | new |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER.md` | new |

No engine/config/strategy/data change this phase. `scratch/` not touched/committed. Unrelated `strategy_research_intake/` dirty files not staged.

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

- **TP-01 regeneration is AUTHORIZED** to proceed in the next phase **using only the official runner** (`research_lab.runners.formal_train_runner`), under the mandatory reconciliation gate.
- **MR-01 remains BLOCKED** until TP-01 is regenerated clean and gate-green.
- The official runner **replaces** `scratch/formal_run_tp01.py` as the only sanctioned, committable mechanism.
- `conservative` multipliers are **owner-approved research/train-only** (NOT real/FTMO/demo/production).

## 13. Remaining Risks / Honest Caveats

- The runner's `execute=True` path is correct-by-construction and lazy-wires the real engine/data/strategies, but — per phase rules — it was **not exercised end-to-end** (no backtest). Full execution validation happens in the next phase, where the reconciliation gate is the hard backstop. This is expected, not hidden.
- `conservative` magnitudes remain laboratory defaults; not market-calibrated. Real/FTMO use requires a separate evidenced phase.
- TP-01 stays, defect-independent, a rejection candidate (PF<1, expectancy<0, 0 trades 2019–2024); this phase only restores infrastructure/governance integrity.

## 14. Next Step

`READY` ⇒ created `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER.md` — regenerate TP-01 train-only 2015–2024 via the official runner, 3 real profiles, mandatory gate, external audit afterward, MR-01 still blocked.

---
*Both gates closed as infrastructure + governance. No backtest, strategy run, optimization, sweep, validation, holdout, news, high-precision or data change was performed.*
