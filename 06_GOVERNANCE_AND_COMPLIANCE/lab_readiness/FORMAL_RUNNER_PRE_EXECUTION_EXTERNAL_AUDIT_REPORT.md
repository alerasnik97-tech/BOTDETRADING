# FORMAL RUNNER PRE-EXECUTION EXTERNAL AUDIT REPORT

**ROLE**: External Institutional Quant Infrastructure Auditor — Formal Backtest Runner Auditor · Python Safety Engineer · Data-Leakage Prevention Officer · Metric Reconciliation Gatekeeper · Cost Model Auditor · Output Policy Auditor · Git Safety Auditor · Pre-Execution Quant Gatekeeper
**DATE**: 2026-05-17
**BASE BRANCH**: `infra/formal-runner-cost-gates-20260517`
**AUDIT BRANCH**: `audit/formal-runner-pre-execution-audit-20260517`
**AUDITED COMMIT**: `08747a3b1bdcb5f55231a69a42f85385289f1dbb` (local == `origin/infra/formal-runner-cost-gates-20260517`)
**SCOPE**: READ CODE → AUDIT CONTRACT → RUN SAFE TESTS → DRY-RUN → STATIC SCAN → REPORT → DECISION. No backtest / strategy run / execute path was exercised.

---

## 1. Status

`FORMAL_RUNNER_BLOCKED_EXECUTE_SIGNATURE_RISK`

The runner is **safe to import and dry-run** and **governance-valid** (it correctly replaces the git-ignored scratch harness), but the `execute=True` path contains **two hard, evidence-backed signature defects** that guarantee a `TypeError` before any backtest artifact or reconciliation could be produced, plus a **co-blocking artifact-write gap** (the runner persists nothing to disk). TP-01 regeneration is **NOT authorized**; a fix phase is required first.

---

## 2. Executive Summary

The honest caveat in `FORMAL_RUNNER_AND_COST_MODEL_GATES_REPORT.md` §13 ("`execute=True` path is correct-by-construction but not exercised end-to-end") was audited statically against the real callee signatures. It is **not** correct-by-construction:

- **B1 — `run_backtest` signature mismatch (BLOCKER).** The runner omits two required positional parameters.
- **B2 — `summarize_result` signature mismatch (BLOCKER).** The runner omits two required positional parameters.
- **B3 — `load_backtest_data_bundle` `data_dirs` type risk.** `tuple[str]` passed where `list[Path]` is annotated (must be verified before execute).
- **B4 — Artifact-write gap (CO-BLOCKER).** `run_single_strategy_formal_train_only` never writes a manifest, config snapshot, summary, trades/equity CSV, or tables to disk. Empirically confirmed: a successful dry-run created **no** output directory.

Everything that is *safe* about the runner passes with high confidence: zero import side-effects, dry-run by default, strict train-only scope, correct 3-profile cost contract with strict monotonicity, a mandatory fail-closed reconciliation gate, a fail-closed CLI (8/8 forbidden inputs rejected), and a clean static scan. The 97/97 green test suite is **real**, but has **zero `execute=True` coverage** — which is precisely why B1/B2 shipped undetected.

**No backtest, strategy run, optimization, sweep, validation, holdout, 2025/2026, news, high-precision, data mutation or code change was performed by this audit.**

---

## 3. Commit Surface Audit

`git show --name-status 08747a3b` → exactly **6 added files**, all in permitted areas:

| File | Verdict |
| :- | :- |
| `03_RESEARCH_LAB/research_lab/runners/__init__.py` | permitted |
| `03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py` | permitted |
| `03_RESEARCH_LAB/research_lab/tests/test_formal_train_runner_contract.py` | permitted |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md` | permitted |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_AND_COST_MODEL_GATES_REPORT.md` | permitted |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER.md` | permitted |

No engine/config/strategy/data/`trades.csv`/`equity_curve.csv`/parquet/ZIP/root/`scratch/`/`local_outputs_do_not_commit/` in the commit. **Commit surface CLEAN.** Working tree has pre-existing **unrelated** dirty files under `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/` (a prior intake audit) — documented, **not touched, not staged**.

---

## 4. Governance Documents Audit

Read: `COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md`, `FORMAL_RUNNER_AND_COST_MODEL_GATES_REPORT.md`, `NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER.md`, `COST_PROFILE_OWNER_DECISION_AND_ROUTING_FIX_REPORT.md`, `TP01_METRIC_FIX_REPORT.md`.

- **Owner cost decision RECORDED**: `COST_MODEL_OWNER_DECISION_RECORDED_RESEARCH_ONLY` — `conservative` ×1.20 spread / ×1.30 slippage **research/train-only**, explicitly **NOT** real/FTMO/demo/production/incubation/deployment; `stress` ×1.35 / ×1.60 unchanged (pre-existing institutional, not invented); commissions 7.0 USD unchanged; strict `base < conservative < stress`.
- **Scratch replaced**: governance unambiguously designates `research_lab.runners.formal_train_runner` as the sole sanctioned, committable, sealable mechanism, replacing the git-ignored `scratch/formal_run_tp01.py`.
- **MR-01 remains BLOCKED** until TP-01 is regenerated clean, gate-green, and externally audited.
- TP-01 regeneration is gated on **this** audit; no holdout/2025-26/validation/optimization/sweep is authorized. Governance chain (metric fix → cost-routing fix → owner ratification → runner) is internally **coherent**.

Governance verdict: **VALID**. The blockers below are engineering defects in the runner, not governance defects.

---

## 5. Runner Contract Audit

| Property | Evidence | Verdict |
| :- | :- | :- |
| Import side-effects | `config.py` imports only `dataclasses`/`pathlib`/`typing`; `metric_reconciliation.py` only `typing`. Runner top-level pulls only those + stdlib. engine/data_loader/strategies/report lazy **inside** `if req.execute`. `__main__` guard present. Subprocess tests 01/30 green; audit dry-run did not import engine/strategies. | **NONE** |
| Dry-run default | `FormalRunRequest.execute=False`; `run_single_strategy_formal_train_only` returns plan when `not req.execute`; CLI prints `[DRY-RUN]`, returns 0 without `--execute`. | **PASS** |
| Train-only scope | `validate_train_only_scope` rejects `holdout`/`sealed_holdout`/`validation` substrings, requires `prepared_train_2015_2024`, rejects `start`/`end ≥ 2025-01-01` and `start>end`; `assert_safe_request` rejects `train_only=False` + holdout/validation/optimization/sweep/high_precision/news flags. | **STRICT** |
| Output policy (validation) | `validate_output_dir` rejects empty / `.zip` / `05_MARKET_DATA_VAULT`/`production`/`incubation`/`scratch` / near-root, requires `…/reports/formal_train_only/…` + a run subdir; `heavy_output_dir` nests `local_outputs_do_not_commit/<profile>`. | **CORRECT (but see §10)** |
| Cost-profile contract | `PROFILE_PLAN` = exactly `base/normal_mode/base`, `conservative/conservative_mode/conservative`, `stress/stress_mode/stress`. `validate_cost_profile_configs` enforces set-equality, mode match, resolved-profile match (mislabel), banned `high_precision_mode`/`precision`/`auto`, duplicate detection, and strict monotonicity. `config.py`: conservative 1.20/1.30, stress 1.35/1.60 → `1.0<1.20<1.35` (spread), `1.0<1.30<1.60` (slippage). | **CORRECT** |
| Reconciliation gate | `seal_run_only_if_reconciled` raises `ReconciliationGateError` on missing manifest, no reconciliation, or any violation code. `reconcile_all(**kwargs)` is signature-compatible with the runner's call. | **ENFORCED** |

---

## 6. Execute Path Audit

**Classification: `EXECUTE_PATH_BLOCKED_SIGNATURE_RISK` (+ artifact-write gap).**

| ID | Severity | Site | Defect | Evidence |
| :- | :- | :- | :- | :- |
| **B1** | BLOCKER | `formal_train_runner.py:328-331` | `run_backtest(strategy_module=, frame=, params=, engine_config=)` omits required positional `news_block: np.ndarray` and `news_filter_used: bool` (no defaults). | `engine.py:597-609`: both params precede `*,` with no default → `TypeError` on first profile. |
| **B2** | BLOCKER | `formal_train_runner.py:332-334` | `summarize_result(result.strategy_name, result.trades, result.equity_curve, result.params, False)` passes 5 positional args. | `report.py:281-293`: requires 7 positional minimum incl. `initial_capital: float` and `selected_score: float \| None` (no defaults) → `TypeError`. |
| **B3** | RISK | `formal_train_runner.py:312-315` | `data_dirs=(req.data_path,)` is `tuple[str]`. | `data_loader.py:359-368` annotates `data_dirs: list[Path]`. Iteration works; `str`-vs-`Path` downstream usage must be verified before execute. |
| **B4** | CO-BLOCKER | `formal_train_runner.py:288-346` | Execute branch builds manifest + reconciliations **in memory** and returns a dict. No `open`/`json.dump`/`to_csv`/`mkdir` anywhere in the module. | Empirical: dry-run with a valid `--output-dir` created **no** directory (`DRY_RUN_DIR_CREATED: False`). No sealed dossier is produced even if B1–B3 are fixed. |

**Correctly wired (verified, not defects):** `STRATEGY_REGISTRY` is keyed by `module.NAME`; `tp01_london_ny_momentum_pullback.NAME == "tp01_london_ny_momentum_pullback"` matches the CLI default; `DEFAULT_PARAMS` + `WARMUP_BARS` exist; `BacktestResult` exposes `strategy_name/trades/equity_curve/params`; `summarize_result` returns a 5-tuple; `summary` carries `profit_factor/expectancy_r/total_return_pct/max_drawdown_pct`; `equity_export["equity"]` exists; `trades_export` is a DataFrame; `reconcile_all` kwargs are compatible.

**Manifest provenance WARNING:** `branch`/`commit` are placeholders — `"(unset)"` in `preflight`, `"(caller-supplied)"` in execute. A sealed run must embed the **real** git branch/commit.

---

## 7. Test Coverage Audit

**Classification: `TEST_COVERAGE_ACCEPTABLE_WITH_WARNINGS`.**

Strong on the safe surface: import-safety (subprocess + AST), dry-run, 3-profile self-report/monotonicity/duplicate/mislabel/high-precision/precision rejection, holdout/sealed_holdout/validation/non-train path rejection, 2025-26 rejection, output policy incl. ZIP/scratch/incubation/data-vault/root, reconciliation blocking (missing manifest / no recon / violations / clean synthetic ledger), manifest no-duplicate/no-empty, CLI fail-closed.

**Critical gap (root cause of B1/B2):** there is **no test that invokes `run_single_strategy_formal_train_only(execute=True)`** with monkeypatched/fake `STRATEGY_REGISTRY` / `load_backtest_data_bundle` / `run_backtest` / `summarize_result`. `test_28` even *asserts* the test module never references `load_backtest_data_bundle`. Consequently the entire execute branch (runner lines 301-346) is untested and the signature defects are invisible to the green suite. Also missing: an artifact-write assertion, a real-branch/commit-in-manifest assertion, and a "one profile fails reconciliation → no seal" test through the actual orchestration.

---

## 8. Dry-Run / CLI Fail-Closed Audit

`$env:PYTHONPATH=…\03_RESEARCH_LAB`; Python 3.14.3; no `--execute` used anywhere.

- **Valid dry-run** (`tp01_london_ny_momentum_pullback`, 2015-01-01→2024-12-31, `prepared_train_2015_2024`): `mode=dry_run`, `executed=False`, 3 correct profiles, heavy dirs nested under `local_outputs_do_not_commit/<profile>`, `reconciliation_required=True`, **exit 0**, **no directory created**.
- **Forbidden inputs — 8/8 fail-closed (exit 2):** `--holdout` / `--validation` / `--optimize` / `--sweep` / `--high-precision` / `--news` → `[FAIL-CLOSED] RunnerSafetyError: forbidden in formal train runner: …`; `--end 2025-01-01` → `2025/2026 data is forbidden`; `--output-dir …RUN_X.zip` → `ZIP output is forbidden`. Valid dry-run → exit 0.

Dry-run / CLI fail-closed: **PASS**. (Dry-run never reaches the broken execute path — the defects are only reachable with `--execute`, which the audit forbids and did not use.)

---

## 9. Static Safety Scan

Tokens `2025|2026|holdout|sealed_holdout|validation|forex_factory|news|high_precision|level2|optimization|sweep|walk_forward|zip|000_PARA_CHATGPT|git add .|local_outputs_do_not_commit|scratch|load_backtest_data_bundle|run_backtest|summarize_result|STRATEGY_REGISTRY` scanned across `runners/`, the contract test, and `lab_readiness/`.

Every hit classified: **intended guard** (date/holdout/zip/scratch/production/incubation rejection logic; banned-mode/profile constants; CLI flags existing only to fail-closed), **negative declaration** (docstrings of what is forbidden; manifest hard-`False` provenance; governance "Safety: NO"), or **lazy execute-only import** (`STRATEGY_REGISTRY`/`load_backtest_data_bundle`/`run_backtest`/`summarize_result` at runner L301-332, after the dry-run return — empirically unreachable on import & dry-run per tests 01/30 and the audit dry-run). No unprotected real use; no `git add .`; no `000_PARA_CHATGPT`. **STATIC SCAN CLEAN — no safety blocker.** (The L301-332 hits are a *correctness* blocker B1/B2, captured in §6, not a safety-scan blocker.)

---

## 10. Output Policy Audit

Path **validation** logic is correct and well-tested (root/data-vault/production/incubation/scratch/ZIP rejection; mandatory `…/reports/formal_train_only/<run>`; heavy artifacts → `local_outputs_do_not_commit/<profile>`; manifest builder rejects duplicate/empty profiles).

**However**, the runner has **no artifact-writing implementation at all** (B4). There is no manifest file, no config/cost snapshot, no `summary.json`, no `trades.csv`/`equity_curve.csv` written under `local_outputs_do_not_commit/`, and no tables. `validate_output_dir`/`heavy_output_dir`/`build_run_manifest` compute and validate **strings**; nothing is persisted. Confirmed empirically (dry-run created nothing). Because the express purpose is to **regenerate the sealed TP-01 train-only dossier**, this is `FORMAL_RUNNER_AUDIT_BLOCKED_ARTIFACT_WRITE_GAP` — a co-blocker independent of B1/B2.

---

## 11. Warnings

- **W1** — Execute path has zero automated coverage (no monkeypatch/fakes). Root cause that let B1/B2 ship green. Must add before any sealed run.
- **W2** — Manifest `branch`/`commit` are placeholders (`"(unset)"`/`"(caller-supplied)"`). A sealed run must embed real git branch/commit.
- **W3** — Seal-time `reconcile_all` is not passed `profiles`, so `reconcile_cost_profiles` (`COST_PROFILE_MISLABEL`/`DUPLICATE`) is **not** part of the gate; cost distinctness relies on `validate_cost_profile_configs` + manifest no-dup. Defensible, but recommend adding profile reconciliation to the seal gate for defence-in-depth.
- **W4** — `load_backtest_data_bundle` `data_dirs` typed `list[Path]`; runner passes `tuple[str]` (B3). Verify/normalize before execute.
- **W5** — Artifact-write gap (B4): no on-disk dossier is produced; fixing B1/B2/B3 alone is insufficient to regenerate TP-01.

---

## 12. Decision

- **TP-01 regeneration is NOT authorized.** A fix phase is mandatory: resolve B1 (`run_backtest` args), B2 (`summarize_result` args), B3 (`data_dirs` type), B4 (write the formal dossier + manifest/snapshot to disk under the validated paths), embed real branch/commit (W2), and add execute-path tests with fakes/monkeypatch (W1); consider W3.
- **A fix is required before any execute.** The official runner is **safe but not execution-ready**.
- **MR-01 remains BLOCKED** until TP-01 is regenerated clean, gate-green and externally re-audited.
- **The official runner correctly replaces `scratch/formal_run_tp01.py`** at the governance/contract level (sole sanctioned, committable, sealable, import-safe, fail-closed mechanism). Governance is valid; only the execute implementation is defective.
- Next prompt: `NEXT_PROMPT_FIX_FORMAL_RUNNER_PRE_EXECUTION_BLOCKERS.md` (fix path, not regenerate).

---

## 13. Safety Verification

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
- code_modified: NO
- force_push: NO
- git_add_dot_used: NO

Audit actions limited to: read code, read docs, run safe unit tests (no backtest), run dry-run/preflight (no `--execute`), static scan, write audit docs. Pre-existing unrelated dirty files were not touched or staged.

---

## 14. Copy-Paste Summary for ChatGPT

```
FORMAL RUNNER PRE-EXECUTION EXTERNAL AUDIT — RESULT
STATUS: FORMAL_RUNNER_BLOCKED_EXECUTE_SIGNATURE_RISK
COMMIT: 08747a3b (infra/formal-runner-cost-gates-20260517); audit branch audit/formal-runner-pre-execution-audit-20260517
SAFE SURFACE: import side-effect-free; dry-run default; train-only scope strict;
  cost profiles base/conservative/stress correct & strictly monotone (1.20/1.30 < 1.35/1.60);
  reconciliation gate fail-closed; CLI fail-closed 8/8; static scan clean; tests 97/97
  (formal_runner 41, cost 11, recon 19, engine 17, stop_entry 3, preflight 6).
BLOCKERS (execute=True path; never run):
  B1 run_backtest call omits required news_block + news_filter_used (engine.py:597-609) -> TypeError
  B2 summarize_result call omits required initial_capital + selected_score (report.py:281-293) -> TypeError
  B3 load_backtest_data_bundle data_dirs tuple[str] vs list[Path] (verify before execute)
  B4 runner writes NO artifacts to disk (dry-run created no dir) -> cannot produce TP-01 dossier
WARNINGS: W1 zero execute-path test coverage (root cause); W2 manifest branch/commit placeholders;
  W3 seal gate omits reconcile_cost_profiles; W4 data_dirs type; W5 artifact-write gap.
DECISION: TP-01 regeneration NOT authorized. Fix required first. MR-01 stays BLOCKED.
  Official runner replaces scratch at governance level but is safe-not-ready for execution.
SAFETY: backtest/strategy/optimization/sweep/validation/holdout/2025-26/news/high_precision/
  data_modified/code_modified/force_push/git_add_dot = ALL NO.
NEXT: NEXT_PROMPT_FIX_FORMAL_RUNNER_PRE_EXECUTION_BLOCKERS.md
```
