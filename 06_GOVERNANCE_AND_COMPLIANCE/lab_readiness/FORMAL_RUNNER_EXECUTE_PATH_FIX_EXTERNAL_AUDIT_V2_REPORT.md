# FORMAL RUNNER EXECUTE PATH FIX EXTERNAL AUDIT V2 REPORT

**ROLE**: External Institutional Formal Runner Auditor · Python Execution-Path Signature Auditor · Artifact Persistence Auditor · Metric Reconciliation Gatekeeper · Data-Leakage Prevention Officer · Cost-Profile Integrity Auditor · Git Safety Auditor · Pre-TP01-Regeneration Gatekeeper
**DATE**: 2026-05-17
**AUDITED BRANCH**: `fix/formal-runner-execute-path-20260517`
**AUDITED COMMIT**: `ba96de4934a66d3938874d725d5fc29800757f52` (local == origin)
**AUDIT BRANCH**: `audit/formal-runner-execute-path-fix-v2-20260517`
**INPUT**: `FORMAL_RUNNER_EXECUTE_PATH_FIX_REPORT.md` (`FORMAL_RUNNER_EXECUTE_PATH_FIX_DONE_READY_FOR_AUDIT_V2`)
**SCOPE**: READ → VERIFY → TEST WITH FAKES → DRY-RUN → STATIC SCAN → REPORT → DECISION. No real backtest / strategy / market data / execute touched.

---

## 1. Status

`FORMAL_RUNNER_EXECUTE_FIX_APPROVED_WITH_WARNINGS_FOR_TP01_REGENERATION`

Every audit-v1 blocker (B1–B4) and warning (W2, W3) is independently verified
closed against the **real source** (not the fix report's prose, not the test
fakes). Signatures re-checked live with `inspect.signature`; the seal-gate→write
ordering re-checked in code; artifact leakage re-checked on disk; 110/110 tests
re-run green. Warnings are minor and non-blocking.

---

## 2. Executive Summary

The fix is **execution-ready**. Independent verification:

- `inspect.signature` on the **real** `engine.run_backtest`, `report.summarize_result`,
  `data_loader.load_backtest_data_bundle` confirms the runner now passes **every**
  required argument (B1/B2/B3) — no missing positional, `data_dirs` is `list[Path]`.
- `config.INITIAL_CAPITAL == 100000.0` confirms B2's capital source is canonical
  (the same constant the engine seeds equity with), not a magic literal.
- Code reading confirms `seal_run_only_if_reconciled(...)` (runner line 522) runs
  **before** `write_run_artifacts(...)` (line 524); a gate failure raises and **no**
  artifact is written — empirically proven by `test_06` (zero `.json`/`.csv`).
- W2: `get_git_identity` uses `subprocess.run([...])` (list argv, **no `shell=True`**,
  `timeout=15`, Git only read), rejects placeholders, fail-closed; dry-run manifest
  shows the **real** branch/commit, resolved dynamically.
- W3: `seal_run_only_if_reconciled` folds `reconcile_all(profiles=…)` into the gate;
  `test_08` proves a mislabel blocks the seal and a clean set does not.
- No output leak: tests write only to auto-cleaned OS tempdirs; the repo working
  tree is clean; no `test_run` / dry-run dir exists in the repo.

**No real backtest, strategy run, optimization, sweep, validation, holdout,
2025-26, news, high-precision, data mutation or code change occurred during this
audit.**

---

## 3. Commit Surface Audit

`git show --name-status ba96de49` → exactly 4 permitted files:
`M runners/formal_train_runner.py` · `A tests/test_formal_train_runner_execute_contract.py`
· `A lab_readiness/FORMAL_RUNNER_EXECUTE_PATH_FIX_REPORT.md` ·
`A lab_readiness/NEXT_PROMPT_AUDIT_FORMAL_RUNNER_EXECUTE_PATH_FIX_V2.md`.
No engine.py / report.py / data_loader.py / strategy / data / CSV / parquet / ZIP
/ root / scratch / local_outputs. **CLEAN.** Pre-existing unrelated
`strategy_research_intake/` dirty files: not touched, not staged.

## 4. Blockers B1–B4 Verification

| ID | Verification (independent) | Verdict |
| :- | :- | :- |
| B1 | `build_disabled_news_block` → `np.zeros(len(frame), dtype=bool)` (numpy lazy); runner passes `news_block` + `news_filter_used=False`; real `run_backtest` required params (live `inspect`) = exactly those passed; engine no-news branch confirmed (engine.py:687-696). | **PASS** |
| B2 | runner passes 7 positionals; real `summarize_result` required params (live `inspect`) = `[…, news_filter_used, initial_capital, selected_score]`; `get_initial_capital` → `config.INITIAL_CAPITAL` (=100000.0, live-checked); 5-tuple return unpacked correctly. | **PASS** |
| B3 | runner passes `[Path(req.data_path)]`; real loader `data_dirs` param confirmed; fake loader asserts `list[Path]`; scope still gated by `validate_train_only_scope`. | **PASS** |
| B4 | `write_run_artifacts` writes manifest/configs/summaries/tables + heavy under `local_outputs_do_not_commit/<profile>`, all under the **re-validated** `req.output_dir`; ZIP refused at writer level; invoked only after seal. `test_05` asserts the full tree + zero `.zip`. | **PASS** |

## 5. W2/W3 Verification

- **W2 — Git identity**: `get_git_identity` = lazy `subprocess.run(["git","-C",root,*args], capture_output=True, text=True, timeout=15)` — list argv, **no `shell=True`**, bounded timeout, Git only read (`rev-parse`), never modified; commit hex-validated; detached → `DETACHED@<sha12>`; `_is_placeholder`/`_require_real_identity` reject `(unset)`/`(caller-supplied)`/empty/null; used in **both** preflight and execute; unavailable Git ⇒ fail-closed `RunnerSafetyError` (`test_11`). Live dry-run manifest: real branch `audit/formal-runner-execute-path-fix-v2-20260517`, real commit `ba96de49…`. **PASS.**
- **W3 — Cost-profile reconciliation in seal**: `seal_run_only_if_reconciled(reconciliations, manifest, profiles=None)`; when `profiles` is passed it adds `mr.reconcile_all(profiles=…)` violations to the gate. Execute path passes `profiles_meta`. `test_08`: mislabel ⇒ `ReconciliationGateError`, clean ⇒ no raise. Backward compatible (legacy 2-arg unaffected ⇒ 41 contract tests green). **PASS.**

## 6. Execute Path Signature Audit

Live `inspect.signature` (audit-run, not trusting report/test):
- `run_backtest` required = `[strategy_module, frame, params, engine_config, news_block, news_filter_used]` ⇒ runner passes all 6. **No mismatch.**
- `summarize_result` required = `[strategy_name, trades, equity_curve, params, news_filter_used, initial_capital, selected_score]` ⇒ runner passes exactly 7 positionals. **No mismatch.**
- `load_backtest_data_bundle` required = `[pair, data_dirs, start, end, execution_mode]` ⇒ runner passes 5 + `target_timeframe="M1"`; `data_dirs=[Path(...)]`. **Compatible.**
- `BacktestResult` fields = `[strategy_name, trades, equity_curve, params, news_filter_used]` ⇒ runner uses the first four. **Correct.** `summarize_result` 5-tuple ⇒ unpacked as 5. **Correct.**

## 7. Fake Execute Test Audit

`test_formal_train_runner_execute_contract.py` (13): 2 real-`inspect.signature`
locks (B1/B2 drift guards) + strict-signature fakes (no `**kwargs`/defaults on
required params ⇒ any runner regression raises) + sys.modules injection of
fake `strategies/data_loader/engine/report` (real data never touched) + `_fake_git`.
Covers: 3 profiles & order, full artifact tree + zero ZIP, recon-failure-blocks-
seal-and-writes (zero files), real branch/commit in manifest, mislabel blocks
seal, no real data loaded, no execute without `execute=True`, Git-unavailable
fail-closed. Restoration of `sys.modules`/package attrs is sentinel-safe.
**Classification: `EXECUTE_COVERAGE_STRONG`** (minor note: no dedicated
`COST_PROFILE_DUPLICATE`-only case — the MISLABEL path + W3 wiring are proven,
so this is a non-blocking warning).

## 8. Artifact Persistence Audit

`write_run_artifacts` re-validates `req.output_dir` (defence-in-depth), writes
under it only; heavy via `heavy_output_dir` → `local_outputs_do_not_commit/<profile>`;
`write_dataframe_csv` refuses `.zip`; `write_json` is UTF-8/sorted/`str`-coerced.
Called strictly after the seal gate. **PASS.**

## 9. Output Policy Audit

`OUTPUT_POLICY_PASS` — manifest `manifests/RUN_MANIFEST.json`; configs
`configs/<p>_ENGINE_CONFIG.json`; summaries `profile_reports/<p>/summary.json`;
tables `profile_reports/<p>/tables/{monthly,yearly}.csv`; heavy
`local_outputs_do_not_commit/<p>/{trades,equity_curve}.csv`; no root / data-vault
/ production / incubation / scratch / ZIP reachable; **no write on reconciliation
failure** (seal line 522 precedes write line 524; `test_06` proves zero files).

## 10. Dry-Run / CLI Fail-Closed Audit

Dry-run (no `--execute`): exit 0, `mode=dry_run executed=False`, real branch/commit
(no placeholder), **no directory created**. Forbidden matrix —
`--holdout/--validation/--optimize/--sweep/--high-precision/--news`,
`--end 2025-01-01`, `…RUN.zip` → **8/8 exit 2**; valid dry-run → exit 0. **PASS.**

## 11. Static Safety Scan

`runners/`, new test, `lab_readiness/` (incl. `shell=True`, `subprocess`,
`run_backtest`, `summarize_result`, `load_backtest_data_bundle`,
`STRATEGY_REGISTRY`). Every hit = intended guard / lazy execute-only / negative
declaration / fake-only-test / benign. Only `shell=True` occurrence is a
negative-declaration docstring; `subprocess.run` uses list argv (no shell);
`high_precision_dir` in the test mirrors the real loader signature (never enables
high precision); `2026` is a fake branch-name string. No real holdout/2025-26/
news/high-precision/ZIP/scratch/data-vault write, no Git-mutating command, no
`git add .`, no unprotected real execute. **CLEAN — no blocker.**

## 12. Test Results

`$env:PYTHONPATH="03_RESEARCH_LAB"` — re-run by this audit:

| Suite | Result |
| :- | :- |
| `test_formal_train_runner_execute_contract.py` | 13/13 OK |
| `test_formal_train_runner_contract.py` | 41/41 OK |
| `test_cost_profiles.py` | 11/11 OK |
| `test_metric_reconciliation.py` | 19/19 OK |
| `test_engine.py` | 17/17 OK |
| `test_engine_stop_entry.py` | 3/3 OK |
| `test_lab_preflight*.py` | 6/6 OK |
| **Total** | **110/110 OK, 0 fail** |

## 13. Warnings

- **W-a** (minor): No dedicated `COST_PROFILE_DUPLICATE`-only test; MISLABEL path
  and W3 gate wiring are proven — non-blocking.
- **W-b** (minor): The runner now hard-requires a Git repo even for dry-run
  (fail-closed by design — correct, but a behavioural change worth noting).
- **W-c** (inherent): The real `load_backtest_data_bundle` / `run_backtest` /
  `summarize_result` chain is exercised only via fakes this phase (by mandate).
  The **first real end-to-end execution** is the upcoming TP-01 regeneration; the
  `inspect`-lock tests + strict fakes mitigate signature risk, but real-data
  behaviour is validated for the first time then. ⇒ the next phase **must** be
  followed by a mandatory external dossier audit.

## 14. Decision

- **TP-01 regeneration is AUTHORIZED (with warnings)** — train-only 2015–2024,
  official runner only, 3 real profiles, mandatory reconciliation+cost-profile
  seal gate, heavy → `local_outputs_do_not_commit`.
- **The official runner is execution-ready** for a single, gated first real run.
- **MR-01 remains BLOCKED** until TP-01 is regenerated clean, gate-green and the
  resulting dossier is externally audited.
- No further runner fix is required to proceed; the next phase is the first real
  execution, then a post-regeneration external audit.

## 15. Safety Verification

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
- code_modified_by_audit: NO
- force_push: NO
- git_add_dot_used: NO

## 16. Copy-Paste Summary for ChatGPT

```
FORMAL RUNNER EXECUTE-PATH FIX — EXTERNAL AUDIT V2
STATUS: FORMAL_RUNNER_EXECUTE_FIX_APPROVED_WITH_WARNINGS_FOR_TP01_REGENERATION
COMMIT: ba96de49 (fix/formal-runner-execute-path-20260517); audit branch
  audit/formal-runner-execute-path-fix-v2-20260517
B1 run_backtest sig: PASS (live inspect: 6 required args all passed; news disabled)
B2 summarize_result sig: PASS (live inspect: 7 required; initial_capital=config.INITIAL_CAPITAL=100000.0; selected_score=None)
B3 data_dirs: PASS (list[Path])
B4 artifact-write: PASS (under re-validated output dir; heavy->local_outputs_do_not_commit; ZIP refused; only post-seal)
W2 git identity: PASS (subprocess list argv, no shell=True, timeout, read-only, fail-closed, no placeholders; real branch/commit live)
W3 cost-profile recon in seal: PASS (reconcile_all(profiles=) folded into gate; mislabel blocks)
TESTS: 110/110 OK (execute 13, contract 41, cost 11, recon 19, engine 17, stop 3, preflight 6)
DRY-RUN: exit 0, real provenance, no dir created. CLI fail-closed 8/8 exit 2.
STATIC SCAN: CLEAN. OUTPUT POLICY: PASS. OUTPUT LEAK: none (tests use temp dirs).
WARNINGS (non-blocking): no duplicate-only cost test; git now required for dry-run;
  first real end-to-end run is the TP-01 regen -> post-regen external audit mandatory.
DECISION: TP-01 regeneration AUTHORIZED (with warnings). Runner execution-ready.
  MR-01 stays BLOCKED until TP-01 clean+gate-green+audited.
SAFETY: real_backtest/strategy/optimization/sweep/validation/holdout/2025-26/news/
  high_precision/data_modified/code_modified_by_audit/force_push/git_add_dot = ALL NO;
  fake_execute_tests_only = YES.
NEXT: NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER_AFTER_AUDIT_V2.md
```
