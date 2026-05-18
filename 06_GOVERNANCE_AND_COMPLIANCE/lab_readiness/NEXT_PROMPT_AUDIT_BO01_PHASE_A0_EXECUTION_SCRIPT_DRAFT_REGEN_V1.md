# NEXT PROMPT AUDIT BO01 PHASE A0 EXECUTION SCRIPT DRAFT REGEN V1

AUTORIZO AUDITORIA EXTERNA READ-ONLY DEL SCRIPT DRAFT PHASE A-0 BO01 TRAIN-ONLY REAL-DATA REGENERADO, SIN EJECUTAR PYTHON, SIN EJECUTAR EL SCRIPT, SIN CARGAR DATOS DE MERCADO, SIN LEER CSV REAL, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.

Actuar como auditor institucional destructivo read-only, Senior Python Security Reviewer, Quant Backtesting Execution Auditor, Data Leakage Auditor, Provenance Auditor, Risk Governance Officer y Git Safety Officer.

## Context

- repo: `alerasnik97-tech/bottrading`
- base branch: `audit/bo01-phase-a0-execution-script-draft-v1-20260518`
- base commit: `25129b169d936e06f51c65c027106cbdd9734bf0`
- regenerated branch: `research/bo01-phase-a0-execution-script-draft-regen-v1-20260518`
- run_id: `BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704`
- script path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704/PHASE_A_EXECUTION_SCRIPT_DRAFT.py`
- manifest path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704/SCRIPT_DRAFT_MANIFEST.json`
- local report path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704/SCRIPT_DRAFT_REPORT.md`
- expected script SHA256: `7bcd55742cf3b9bee46c66572d7f2163f8a4248acccfa5133ffa17d61e30ee15`
- provenance_model: `BASE_COMMIT_PLUS_SCRIPT_HASH`

## Hard Prohibitions

- no Python execution;
- no script execution;
- no script import;
- no py_compile;
- no tests;
- no notebooks;
- no data loading;
- no CSV read;
- no M5/M15 read;
- no `05_MARKET_DATA_VAULT` content read;
- no backtest;
- no formal train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep/grid search/walk-forward/parameter search;
- no code/test/data/runner changes;
- no script/manifest/local report modification;
- no `git add .`;
- no reset/rebase/clean/stash;
- no force push;
- no demo/real/FTMO;
- no edge/profitability claims.

## Audit Checks

Verify read-only:

- local script exists;
- manifest exists;
- local report exists;
- script SHA256 matches manifest and governance report;
- output root gitignored;
- local artifacts not tracked;
- no Python;
- no script execution;
- no data loaded;
- no CSV read;
- no backtest;
- provenance model is non-circular;
- manifest has `generated_from_base_commit = 25129b169d936e06f51c65c027106cbdd9734bf0`;
- manifest has no ambiguous `generated_from_commit`;
- final governance commit is verified by Git, not expected inside manifest;
- script content safety;
- activation/hash gate;
- path authorization;
- data proof logic;
- runner gate;
- strategy gate;
- output policy;
- cost profiles;
- no optimization/sweep;
- no demo/real/FTMO;
- no secrets;
- no destructive commands;
- no unsupported paths;
- Phase A-1 blocked until script audit passes.

## Diff Scope

The regenerated branch must commit only:

- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_V1.md`

Block if script, manifest, local report, local output root, CSV, data, tests, runner, strategy, data vault, ZIP, notebook, or root scratch files are tracked.

## Static Safety Scan

Use textual scan over the script, manifest, local report, governance report and this next audit prompt for:

`validation|holdout|2025|2026|optimization|sweep|grid search|walk-forward|parameter search|champion|FTMO|demo|real|edge|profitability|rentabilidad|estrategia rentable|backtest|train|PnL|PF|profit factor|winrate|drawdown|Sharpe|Sortino|expectancy|equity curve|git add \.|reset --hard|rebase|git clean|git stash|force push|Remove-Item|rm -rf|del /|rmdir|shutil.rmtree|os.remove|unlink|subprocess|os.system|eval|exec|pickle|requests|curl|wget|Invoke-WebRequest|token|password|secret|api_key|credential|broker|telegram|perfect|perfectly|flawless|flawlessly|100%|certified|guaranteed|sealed|locked|robust|secured|successfully`

Classify hits as:

- NEGATIVE_DECLARATION_OK
- GOVERNANCE_TERM_OK
- FUTURE_SCRIPT_TERM_OK
- SCRIPT_REQUIRED_LOGIC_OK
- PROVENANCE_MODEL_OK
- SECURITY_SCAN_OK
- LANGUAGE_WARNING
- BLOCKER

Rules:

- `pd.read_csv` permitted only inside future strict loader, not executed;
- validation/holdout/2025/2026 permitted only as guard/prohibition;
- demo/real/FTMO permitted only as prohibition;
- edge/profitability/rentabilidad permitted only as negation;
- performance terms permitted only as future Phase A-1 metrics policy;
- destructive/security terms permitted only as scan/prohibition or `git check-ignore`;
- `subprocess` permitted only when limited to `git check-ignore`;
- `exec_module` permitted only for fixed-path runner/strategy import;
- any real secret is blocker.

## Decision Values

Choose exactly one:

- `BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_AUDIT_PASS_READY_FOR_PHASE_A1_OWNER_DECISION`
- `BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_AUDIT_PASS_WITH_WARNINGS`
- `AUDIT_BLOCKED_REGEN_SCRIPT_HASH_MISMATCH`
- `AUDIT_BLOCKED_REGEN_MANIFEST_INCONSISTENT`
- `AUDIT_BLOCKED_REGEN_PROVENANCE_AMBIGUOUS`
- `AUDIT_BLOCKED_REGEN_SCRIPT_EXECUTED_OR_DATA_LOADED`
- `AUDIT_BLOCKED_REGEN_DIFF_SCOPE`
- `AUDIT_BLOCKED_REGEN_SCRIPT_SAFETY`
- `AUDIT_BLOCKED_REGEN_STATIC_SAFETY_SCAN`

If PASS or PASS_WITH_WARNINGS, create a next owner decision prompt for Phase A-1 with exact activation phrase and hash verification. If BLOCKED, create a blocker-fix prompt requiring a new Phase A-0 package; do not patch the audited package manually.
