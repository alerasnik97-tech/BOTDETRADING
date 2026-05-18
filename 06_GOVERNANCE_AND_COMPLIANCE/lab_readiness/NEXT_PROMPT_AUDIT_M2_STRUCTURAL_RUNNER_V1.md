# NEXT PROMPT — AUDIT M2 STRUCTURAL RUNNER V1

This prompt is to be executed in **READ-ONLY AUDIT MODE**.
Under blocker penalty, the following are strictly **PROHIBITED** during this audit:
- NO executing Python scripts or commands.
- NO executing helper scripts (such as `safety_scan.py`).
- NO M2 execution.
- NO loading of market data.
- NO modifying strategy code, engine, runner, or test files.
- NO backtesting or formal training.
- NO validation or holdout partition access.
- NO 2025 or 2026 data loading.
- NO optimization sweeps, grid searches, or walk-forward parameters.

---

## 1. Audit Objective
Verify the implementation, contract, safety boundaries, and synthetic test results of the newly created M2 Structural Runner.

---

## 2. Verification Checklist

### 2.1 File Scope and Whitelist Verification
- Confirm that the base branch is `research/m2-conservative-structural-bo01-mr02-v1-20260518`.
- Confirm that the implementation branch is `research/m2-structural-runner-bo01-mr02-v2-20260518` (as active branch).
- Verify the physical local HEAD SHA is descendiente of commit `65304d76a6ffda1eb09f971fad89b2f2b4692cf9`.
- Verify that ONLY the whitelisted 5 files are modified/created:
  - `03_RESEARCH_LAB/research_lab/runners/m2_structural_runner.py`
  - `03_RESEARCH_LAB/research_lab/tests/test_m2_structural_runner_contract.py`
  - `03_RESEARCH_LAB/research_lab/tests/test_m2_structural_runner_safety.py`
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M2_STRUCTURAL_RUNNER_IMPLEMENTATION_REPORT_V1.md`
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M2_STRUCTURAL_RUNNER_V1.md`
- Verify that **no** other code, tests, or market data files were modified.

### 2.2 Runner Code and Contract Safety
- Verify that `m2_structural_runner.py` does NOT contain active performance calculations (PnL, Sharpe, winrate, Profit Factor, etc.).
- Verify that `validate_frame_for_m2` enforces:
  - Timezone-aware DatetimeIndex.
  - No 2025 or 2026 years.
  - No validation or holdout partition keywords.
  - Presence of standard OHLC columns.
  - Approximate M5 cadence and absence of NaN values.
- Verify that `run_structural_counts` executes signal calculations safely without generating trades lists or return vectors.

### 2.3 Synthetic Test Verification
- Verify that both test files are standard `unittest.TestCase` modules.
- Confirm that the tests use ONLY synthetic frames created in setUp (no CSV files loaded, no data vault path accessed).
- Verify that all tests pass without errors or warnings.

---

## 3. Allowed Methods
The auditor is permitted to use **ONLY** read-only text commands:
- Git inspection commands (`git status`, `git branch`, `git log`, `git diff`).
- Text search commands (`rg` or native PowerShell search commands).
- Reading markdown files using file viewers.

---

## 4. Final Audit Decision
The auditor must report a final safety status:
- **STATUS = PASS:** If all checks comply, the runner code is safe, tests pass, only whitelisted files are modified, and no performance terms logic is active.
- **STATUS = BLOCKER:** If any python script execution is allowed, any performance metrics are permitted, or any execution is possible without the exact activation gate phrase.
