# SUBBATCH 1A BO01/MR02 MICRO-RUN PROTOCOL DESIGN V1

## 1. Status
**`MICRO_RUN_PROTOCOL_DESIGN_READY_FOR_EXTERNAL_AUDIT`**

---

## 2. Nature Of This Document
This is a design-only document. It does NOT authorize any code execution, micro-runs, dry-runs, backtests, validation unsealings, or parameter sweeps. No commands are executed during this phase. No micro-run, dry-run, backtest, train, validation, holdout, or 2025/2026 data is authorized or exposed. No optimization or parameter sweep is authorized.

---

## 3. Purpose
This protocol defines how a future, minimal, safe, and controlled micro-run execution would be conducted solely to verify:
- import wiring of strategy candidates;
- signal function call path;
- fail-closed behavior on invalid fixtures;
- output policy compliance;
- absence of data leakage;
- absence of unauthorized files in the tree;
- lack of runner or engine mutation;
- lack of strategy registry mutation during execution.

**This protocol cannot prove edge, profitability, robustness, or readiness for real/demo/FTMO.** It is strictly a technical verification plumbing design.

---

## 4. Candidates
- **BO01** (London Breakout)
- **MR02** (London Fakeout Reversion)

### Current State
- Existing files are limited to strategy skeletons and targeted unit/contract tests.
- Blocker patch has been externally audited.
- No performance or edge exists or is asserted.
- No dynamic execution has been initiated.

---

## 5. Preconditions Before Any Future Execution
Before any future micro-run execution is scheduled, all of the following hard gates must be met:
1. This design protocol must pass external read-only audit.
2. Owner must explicitly approve future execution with a separate exact phrase.
3. Worktree must be clean or W-01 dirty files must be quarantined with documented rationale.
4. W-02 output policy gate must be defined and audited.
5. No active Python runner/backtest/optimization process.
6. No holdout (2025/2026) exposure.
7. No 2025/2026 data used.
8. No validation data used.
9. No optimization or sweep command executed.
10. No Sub-Batch 1B strategy candidates included.
11. One single writer agent maximum in the laboratory.
12. Output destination folder must be pre-approved.
13. No `git add .` or `git add --all` allowed.
14. No output files created in the repository root.
15. No `trades.csv` or `equity_curve.csv` files committed.
16. No ZIP files created or modified.

---

## 6. Future Data Policy
No data is loaded or read in this design-only phase. Future execution paths are restricted to:

### PHASE M0 — Synthetic-only controlled micro-run:
- Synthetic M5 bar fixtures only.
- No access to `05_MARKET_DATA_VAULT`.
- No historical market data.
- No 2025/2026 data.
- No validation or holdout data.
- Purpose: Technical plumbing verification only.

### Optional Later Path (PHASE M1 — Tiny train-only controlled slice):
- Strictly train-only slice (2015–2024).
- No validation data, no holdout data, no 2025/2026 data.
- No optimization or sweep.
- No performance inference.
- Authorized only if this design protocol audit is green and a separate owner approval is granted.

---

## 7. Future Execution Scope
If later approved, the future micro-run execution may test ONLY:
- Candidate imports (`BO01Strategy`, `MR02Strategy`).
- Call to `default_params`.
- Signal function execution on a controlled synthetic fixture.
- Correct zero-signal behavior outside defined session hours.
- Immediate fail-closed throw on malformed or incomplete synthetic fixtures.
- Storing outputs strictly inside the designated folder.
- Absence of unexpected file creation in other folders.

### The future micro-run MUST NOT test:
- Profit factor (PF), win rate, or drawdown.
- Sharpe ratio or expectancy.
- Regime robustness.
- Parameter sweeps or parameter search.
- Model selection or leaderboard ranking.

---

## 8. Future Command Policy
The following command templates are drafts only. **DO NOT RUN ANY COMMANDS IN THIS PHASE.**

Every command is marked as:
`DRAFT_DO_NOT_RUN — for future owner-approved execution only.`

### Draft Commands:
1. `DRAFT_DO_NOT_RUN` — Check git status:
   `git status --short`
2. `DRAFT_DO_NOT_RUN` — Verify active Python processes:
   `Get-Process python -ErrorAction SilentlyContinue`
3. `DRAFT_DO_NOT_RUN` — Run targeted contract tests (plumbing verification):
   `pytest 03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_bo01.py`
4. `DRAFT_DO_NOT_RUN` — Synthetic plumbing dry-run (no --execute, no data vault):
   `python -m research_lab.runners.synthetic_plumbing_runner --strategy BO01 --synthetic-only`

### Forbidden Commands:
- `python -m research_lab.runners.formal_train_runner --execute`
- Any backtest commands using real market data.
- Any validation or holdout commands.
- Any parameter optimization or sweep commands.

---

## 9. Future Output Policy
Future execution outputs must be strictly routed to:
`03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/microrun_subbatch_1a/<RUN_ID>/`

### Hard Rules:
- The `local_outputs_do_not_commit/` directory must be registered in `.gitignore` or execution must immediately abort.
- No outputs are permitted in the repository root or `05_MARKET_DATA_VAULT`.
- No ZIP files are permitted in the repo.
- No `trades.csv` or `equity_curve.csv` files may be staged or committed.
- No screenshot artifacts are permitted unless explicitly approved.
- No large files (>1MB) may be created.
- No outputs may be uploaded to GitHub.
- An output manifest listing all created files is required.
- No pre-existing outputs are cleaned during this design phase.

---

## 10. Future Abort Conditions
A future micro-run execution must immediately abort if:
- The current git branch is `main`.
- Pre-existing dirty tree files (W-01) are modified or untracked in other directories.
- Staged changes exist before starting.
- Active Python research, backtesting, or optimization processes are running.
- The designated output directory is not properly ignored in `.gitignore`.
- Any command attempts to read from or write to `05_MARKET_DATA_VAULT`.
- Any command attempts to load 2025/2026 data.
- Any command accesses validation or holdout datasets.
- Any command attempts a sealed train, validation, optimization, or parameter sweep.
- Any file is written in the repo root or outside the approved path.
- Any execution requires changes to the core engine, runner, or data loader.

---

## 11. Future Success Criteria
Future micro-run success, if approved, may only signify:
- Candidate strategy files imported correctly.
- Signal calculations completed on synthetic fixtures without exceptions.
- Fail-closed logic successfully blocked invalid inputs.
- Outputs remained confined to the pre-approved folder.
- No untracked, heavy, or forbidden files were created.
- No forbidden datasets were touched.

**It does NOT signify that the strategy has an edge, is profitable, is ready for backtesting, training, validation, or live/demo/FTMO deployment.**

---

## 12. Future Report Requirements
Any future execution report must include:
- The exact owner approval phrase.
- Branch and commit SHA.
- Complete log of commands executed.
- List of files created (output manifest).
- Formal declaration of data source used (confirming no holdout, no validation, no 2025/2026, no optimization).
- Target verdict: Ready for external read-only audit only.

---

## 13. External Audit Requirement
- **Before execution:** This protocol design document must pass external read-only audit.
- **After execution:** Any future micro-run execution results must pass a separate read-only audit before any other phase can be authorized.

---

## 14. Allowed Next Step
- External read-only audit of this micro-run protocol design.

---

## 15. Forbidden Next Steps
- NO immediate micro-run preflights or dynamic executions are authorized.
- NO dry-runs, parameter sweeps, or optimization sweeps are permitted.
- NO sealed train backtests on 2015-2024 train data are allowed.
- NO validation unsealing or holdout (2025/2026) exposure is permitted.
- NO parallel writing agents are permitted in the laboratory.
- NO use of production, demo, real, or FTMO accounts is allowed.

---
*End of Design Protocol*
