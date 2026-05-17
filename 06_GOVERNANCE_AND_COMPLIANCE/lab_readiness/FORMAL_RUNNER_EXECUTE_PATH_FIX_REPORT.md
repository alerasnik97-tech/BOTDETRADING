# FORMAL RUNNER EXECUTE PATH FIX REPORT

**ROLE**: Institutional Research Infrastructure Engineer · Formal Runner Repair Engineer · Python Signature Compatibility Auditor · Artifact Persistence Architect · Metric Reconciliation Gatekeeper · Backtest Safety Officer · Git Safety Officer · Pre-Execution Quant Gatekeeper
**DATE**: 2026-05-17
**BASE BRANCH**: `audit/formal-runner-pre-execution-audit-20260517` @ `1cccdcc0`
**FIX BRANCH**: `fix/formal-runner-execute-path-20260517`
**INPUT AUDIT**: `FORMAL_RUNNER_BLOCKED_EXECUTE_SIGNATURE_RISK` (`FORMAL_RUNNER_PRE_EXECUTION_EXTERNAL_AUDIT_REPORT.md`)

---

## 1. Status

`FORMAL_RUNNER_EXECUTE_PATH_FIX_DONE_READY_FOR_AUDIT_V2`

All four blockers (B1–B4) and both warnings (W2, W3) are closed. The `execute=True`
path is now signature-correct, persists the formal dossier under the validated
output area, embeds real Git provenance, and is covered by a hermetic fakes-only
contract suite. **110/110 tests green, dry-run safe, CLI fail-closed 8/8, static
scan clean. No real backtest / strategy / market data was touched.**

---

## 2. Executive Summary

The audit proved the documented "execute path is correct-by-construction" claim
false: `run_backtest` / `summarize_result` were called with missing required
positional args (guaranteed `TypeError`), `data_dirs` had the wrong type, no
artifact was ever written, and the manifest carried `(unset)` / `(caller-supplied)`
placeholders. Each defect was reproduced by reading the **real** callee signatures
in source (engine.py:597-609, report.py:281-293, data_loader.py:359-368, the
`config.INITIAL_CAPITAL` constant, engine.py:684/687-696 for the no-news branch),
fixed minimally in the orchestration layer only (no engine/strategy/signal logic
touched), and locked with tests — including two `inspect.signature` lock tests
against the real engine/report so future drift fails fast.

---

## 3. Blockers Fixed

### B1 — `run_backtest` signature
`engine.run_backtest(strategy_module, frame, params, engine_config, news_block, news_filter_used, *, …)` — `news_block` and `news_filter_used` are **required** (no defaults). Added `build_disabled_news_block(frame)` → `np.zeros(len(frame), dtype=bool)` (NumPy lazily imported inside the helper; module import stays NumPy-free). The execute call now passes `news_block=<all-False>` and `news_filter_used=False`; verified against the engine no-news branch (`engine.py:687-696`: `news_events is None` → `np.asarray(news_block, dtype=bool)`). News is **never** enabled; no calendar/forex_factory IO. Locked by `test_real_run_backtest_requires_news_args` + the strict-signature fake.

### B2 — `summarize_result` signature
`report.summarize_result(strategy_name, trades, equity_curve, params, news_filter_used, initial_capital, selected_score, …)` requires 7 positionals. The call now passes `news_filter_used=False` (the real 5th positional — the prompt's "optimization_run" label does not exist in code; **source was followed, not the prompt**), `initial_capital=get_initial_capital(base_cfg)` and `selected_score=None`. `get_initial_capital` returns `config.INITIAL_CAPITAL` (=100000.0) — the canonical constant the engine itself seeds equity with (`engine.py:684`); no magic literal invented. Monthly/yearly exports are now captured (4th/5th return values). Locked by `test_real_summarize_result_requires_capital_and_score` + the strict-signature fake.

### B3 — `data_dirs` type
Loader signature is `data_dirs: list[Path]`. The call now passes `[Path(req.data_path)]` (`Path` imported at module top-level). Fake loader asserts `isinstance(data_dirs, list)` and every element is a `Path`.

### B4 — Artifact write gap
Added `write_json` (UTF-8, indent=2, sorted, `str`-coerced, `mkdir parents`), `write_dataframe_csv` (ZIP-refusing, `mkdir parents`) and `write_run_artifacts`. After the seal gate passes, the dossier is written **only** under the re-validated `req.output_dir`:
`manifests/RUN_MANIFEST.json` · `configs/<profile>_ENGINE_CONFIG.json` · `profile_reports/<profile>/summary.json` · `profile_reports/<profile>/tables/{monthly,yearly}.csv` · `local_outputs_do_not_commit/<profile>/{trades,equity_curve}.csv`. No root / data-vault / production / incubation / scratch / ZIP write is reachable. Empirically: a dry-run still creates **no** directory; an executed fake run creates exactly the above and **zero** `.zip`.

### W2 — Real branch/commit
Added `get_git_identity()` (lazy `subprocess`, **list argv — no `shell=True`**, `timeout=15`, Git only read, never modified). Returns the real `(branch, commit)`; detached HEAD → `DETACHED@<sha12>` (real provenance, never a placeholder). `_is_placeholder` / `_require_real_identity` reject `(unset)`/`(caller-supplied)`/empty/null. Used in **both** `preflight` and the execute path; unavailable Git ⇒ fail-closed `RunnerSafetyError`. Confirmed live: dry-run manifest now shows `branch=fix/formal-runner-execute-path-20260517`, `commit=1cccdcc0…`.

### W3 — Cost-profile reconciliation in the seal gate
`seal_run_only_if_reconciled` gained an optional `profiles=` kwarg; when provided it folds `metric_reconciliation.reconcile_all(profiles=…)` (`COST_PROFILE_MISLABEL` / `COST_PROFILE_DUPLICATE`) into the gate. The execute path passes `profiles_meta` (`{name: {cost_profile, execution_mode}}`). Backward compatible: the legacy 2-arg form is unchanged, so the 41 existing contract tests stay green. `test_08` proves a mislabelled profile blocks the seal and a clean set does not.

---

## 4. Execute Path Contract

`preflight` (offline validation + real-provenance manifest) → on `execute=True`:
lazy-import heavy modules → validate cost profiles → resolve Git identity →
load bundle (`[Path]`, M1, normal_mode) → per profile {`run_backtest`(news off) →
`summarize_result`(real capital) → per-profile `reconcile_all`} → seal gate
(per-profile **and** cost-profile reconciliation; refuses placeholder-free manifest
absence) → **only then** `write_run_artifacts`. Any failure raises before sealing
and before any artifact is written.

## 5. Artifact Persistence Contract

All writes are confined to the `validate_output_dir`-validated `req.output_dir`
(re-validated inside `write_run_artifacts`). Heavy artifacts route exclusively to
`local_outputs_do_not_commit/<profile>`. ZIP is refused at the writer level too.
Nothing is written on dry-run, on scope/policy rejection, or on reconciliation
failure.

## 6. Reconciliation Gate Contract

Seal is refused on: missing manifest, no reconciliation performed, any per-profile
violation code, or (W3) any cost-profile mislabel/duplicate. Artifacts are written
strictly **after** the gate passes (`test_06`: contradictory summary ⇒
`ReconciliationGateError`, zero files written).

## 7. Tests

`$env:PYTHONPATH="03_RESEARCH_LAB"`

| Suite | Result |
| :- | :- |
| `test_formal_train_runner_execute_contract.py` (**new**, fakes only) | **13/13 OK** |
| `test_formal_train_runner_contract.py` | 41/41 OK |
| `test_cost_profiles.py` | 11/11 OK |
| `test_metric_reconciliation.py` | 19/19 OK |
| `test_engine.py` | 17/17 OK |
| `test_engine_stop_entry.py` | 3/3 OK |
| `test_lab_preflight*.py` | 6/6 OK |
| **Total** | **110 pass, 0 fail** (+13 vs. 97; no regression) |

New suite covers: real `run_backtest`/`summarize_result` signatures (lock + strict
fakes), `data_dirs` → `list[Path]`, exactly 3 profiles, artifact writes under a
valid tempdir output (no ZIP), reconciliation-failure-blocks-seal-and-writes, real
branch/commit in manifest, cost-profile mislabel blocks seal, no real market data
loaded, no execute without `execute=True`, and Git-identity-unavailable fail-closed.

## 8. Dry-Run / CLI Fail-Closed

- Dry-run (no `--execute`): exit 0, `executed=False`, 3 correct profiles, **real**
  branch/commit in manifest, **no directory created**.
- Forbidden inputs `--holdout/--validation/--optimize/--sweep/--high-precision/--news`,
  `--end 2025-01-01`, `…RUN_X.zip` → **8/8 `[FAIL-CLOSED]` exit 2**, raised before
  Git/backtest; valid dry-run → exit 0.

## 9. Static Safety Scan

`runners/`, new tests, `lab_readiness/` scanned (incl. `shell=True`, `subprocess`).
Every hit classified **intended guard** (date/holdout/zip/scratch rejection;
`local_outputs_do_not_commit` routing; banned-mode constants), **negative
declaration** (docstrings: "no `shell=True`", "No calendar / forex_factory / news
IO", manifest hard-`False`), or **benign** (`subprocess` = read-only
`get_git_identity`: list argv, no shell, timeout, Git never modified; fakes/tempdir
in the test). No `git add .`, no `000_PARA_CHATGPT`, no real `shell=True`, no
`walk_forward`/`level2`. **No safety violation.**

## 10. Safety Verification

- real_backtest_run: NO
- real_strategy_run: NO
- fake_execute_tests_only: YES
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

(Engine / signal logic / TP-01 / MR-01 strategy code / data vault: untouched. Only
`runners/formal_train_runner.py` + a new test + 2 governance docs changed.)

## 11. Remaining Risks

- A **real** `execute=True` run is still **NOT authorized**: it requires external
  pre-execution audit **v2**. This phase validates the path only with fakes
  (no real data/backtest by rule).
- **TP-01 regeneration remains NOT authorized** until audit v2 passes.
- **MR-01 remains BLOCKED** until TP-01 is regenerated clean, gate-green and
  externally audited.
- `get_git_identity` depends on Git being available (fail-closed by design).
- The real `load_backtest_data_bundle` is type-consistent with `[Path]` but not
  exercised end-to-end here (by rule); audit v2 + the operator phase validate it.
- TP-01 stays, defect-independent, a rejection candidate (PF<1, expectancy<0);
  this phase restores runner executability, not strategy edge.

## 12. Next Step

`READY` ⇒ created `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_FORMAL_RUNNER_EXECUTE_PATH_FIX_V2.md`
— external pre-execution audit v2 (no real backtest; fakes only) must approve
before any real TP-01 regeneration.

---
*Execute-path repaired in the orchestration layer only; signatures verified against
source, artifacts persisted under the validated area, real provenance enforced. No
backtest, strategy run, optimization, sweep, validation, holdout, 2025-26, news,
high-precision or data change was performed.*
