# NEXT PROMPT — AUDIT M2 STRUCTURAL RUNNER WARNING PATCH V1

This prompt is to be executed in **READ-ONLY AUDIT MODE**.
Under blocker penalty, the following are strictly **PROHIBITED** during this audit:
- NO executing Python scripts with real market data.
- NO loading of market data.
- NO M2 execution.
- NO modifying strategy code, engine, runner, or test files.
- NO backtesting or formal training.
- NO validation or holdout partition access.
- NO 2025 or 2026 data loading.
- NO optimization sweeps or parameters search.

---

## 1. Audit Objective
Verify the implementation of the M2 Structural Runner Warning Patch.

---

## 2. Verification Checklist

### 2.1 File Scope and Whitelist Verification
- Confirm that the base branch is `audit/m2-structural-runner-bo01-mr02-v1-20260518` at commit `fc7671bd6e409337cbfbd470924213d08ad2fb61`.
- Confirm that the implementation branch is `research/m2-structural-runner-warning-patch-v1-20260518` (active branch).
- Verify the physical local HEAD SHA is descendant of commit `fc7671bd6e409337cbfbd470924213d08ad2fb61`.
- Verify that ONLY the whitelisted 5 files are modified/created:
  - `03_RESEARCH_LAB/research_lab/runners/m2_structural_runner.py`
  - `03_RESEARCH_LAB/research_lab/tests/test_m2_structural_runner_contract.py`
  - `03_RESEARCH_LAB/research_lab/tests/test_m2_structural_runner_safety.py`
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M2_STRUCTURAL_RUNNER_WARNING_PATCH_REPORT_V1.md`
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M2_STRUCTURAL_RUNNER_WARNING_PATCH_V1.md`

### 2.2 W-01 Correctness (valid_signal_count semantics)
- Verify in `m2_structural_runner.py` that `counts["valid_signal_count"]` is incremented ONLY inside the try block after all contract checks (non-dict, signal in (1,-1), direction in ("long","short")) pass.
- Verify that if a non-None signal fails contract checks, both `valid_signal_count` and `contract_valid_count` remain unchanged, while `exception_count` and `fail_closed_count` are correctly incremented.

### 2.3 W-02 Correctness (Safety Tests)
- Verify in `test_m2_structural_runner_safety.py` that `test_no_forbidden_active_logic_terms` uses `inspect.getsource(m2_structural_runner)` to read the module source.
- Verify that explicit contract assertions are added for `"read_csv"`, `"to_csv"`, `"05_MARKET_DATA_VAULT"`, and `"EURUSD_M5"` to guarantee they are absent from active logic.

### 2.4 Synthetic Test Verification
- Verify that `test_m2_structural_runner_contract.py` defines `MalformedStrategyBO01` returning `{"direction": "long"}` with missing `signal` key.
- Verify that `test_malformed_signal_is_fail_closed_not_valid` successfully checks fail-closed behavior for malformed signals.
- Verify that `test_valid_signal_count_equals_contract_valid_count_for_valid_signals` successfully verifies counts equality.
- Confirm all 39 tests pass without errors or warnings.

---

## 3. Allowed Methods
The auditor is permitted to use **ONLY** read-only text commands:
- Git inspection commands (`git status`, `git branch`, `git log`, `git diff`).
- Text search commands (`rg` or native PowerShell search commands).
- Reading markdown files using file viewers.
- Executing unit tests in-memory to confirm passes.

---

## 4. Final Audit Decision
The auditor must report a final safety status:
- **STATUS = PASS:** If all checks comply, warnings are resolved, tests pass, and no performance terms logic is active.
- **STATUS = BLOCKER:** If any python script execution is allowed, any performance metrics are permitted, or any execution is possible without the exact activation gate phrase.
