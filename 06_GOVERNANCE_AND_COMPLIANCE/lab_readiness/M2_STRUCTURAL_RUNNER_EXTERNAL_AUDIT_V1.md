# M2 STRUCTURAL RUNNER EXTERNAL AUDIT V1

## 1. Audit Status

**`M2_STRUCTURAL_RUNNER_AUDIT_PASS_WITH_WARNINGS`**

No blockers found. Two minor warnings documented. Runner is eligible for owner decision on M2 retry.

---

## 2. Executive Verdict

The runner was created within the authorized whitelist scope. No performance metrics are computed anywhere in active logic. No real data is read or written. No data vault paths are referenced. Tests are 100% synthetic and pass cleanly. The runner contract is structurally compatible with BO01 and MR02 signal outputs.

Two warnings are documented:
- **W-01**: `run_structural_counts` increments `valid_signal_count` before verifying the minimum signal contract. If contract verification raises an exception (e.g., missing `signal` key), the bar is counted in `valid_signal_count` but then moved to `exception_count`/`fail_closed_count`, resulting in a transient overcounting of `valid_signal_count` versus `contract_valid_count`. This is not a blocker — the final counts reflect the correct divergence — but the semantics of `valid_signal_count` are slightly overloaded.
- **W-02**: `test_no_forbidden_active_logic_terms` uses `open()` to read the runner's own source file and confirms the `FORBIDDEN_PERFORMANCE_TERMS` list exists. This is a shallow static check — it verifies the list is present but does not verify that the forbidden terms are absent from active logic paths. Acceptable as a complement to the full static scan, but should not be considered a complete contract guard on its own.

This audit does NOT certify BO01/MR02 as having edge, profitability, or robustness. It does NOT authorize validation, holdout, or 2025/2026 data use. It does NOT constitute a formal backtest or training run.

---

## 3. Scope Audited

- **Audit branch**: `audit/m2-structural-runner-bo01-mr02-v1-20260518`
- **Implementation branch audited**: `research/m2-structural-runner-bo01-mr02-v2-20260518`
- **Commit audited**: `ac34c8a82bd44bc18bd17600385687efbe48d7b6`
- **Base commit**: `65304d76a6ffda1eb09f971fad89b2f2b4692cf9`
- **Files inspected**:
  1. `03_RESEARCH_LAB/research_lab/runners/m2_structural_runner.py`
  2. `03_RESEARCH_LAB/research_lab/tests/test_m2_structural_runner_contract.py`
  3. `03_RESEARCH_LAB/research_lab/tests/test_m2_structural_runner_safety.py`
  4. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M2_STRUCTURAL_RUNNER_IMPLEMENTATION_REPORT_V1.md`
  5. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M2_STRUCTURAL_RUNNER_V1.md`
- **Tests executed**: synthetic only (10 contract + 4 safety + 11 BO01 + 12 MR02 = 37 total)
- **M2 executed**: NO
- **Real data loaded**: NO

---

## 4. Safety Verification

| Check | Result |
|---|---|
| code_modified_by_audit | NO |
| tests_modified_by_audit | NO |
| data_modified | NO |
| data_loaded_by_audit | NO |
| M2_executed | NO |
| backtest | NO |
| train | NO |
| validation | NO |
| holdout | NO |
| 2025/2026_used | NO |
| optimization/sweep | NO |
| git_add_dot | NO |
| reset/rebase/clean/stash | NO |
| force_push | NO |

---

## 5. Diff Scope Audit

**Result: PASS_DIFF_SCOPE_WHITELIST_ONLY**

`git diff --name-status 65304d76..HEAD` shows exactly 5 files, all `A` (Added), all within the authorized whitelist. No modifications to:
- BO01Strategy.py ✓ untouched
- MR02Strategy.py ✓ untouched
- engine ✓ untouched
- data_loader ✓ untouched
- factory ✓ untouched
- strategies/__init__.py ✓ untouched
- 05_MARKET_DATA_VAULT ✓ untouched
- Any root file, notebook, or legacy output ✓ untouched

---

## 6. Runner Code Audit

**Result: PASS_RUNNER_CODE_SAFE (with W-01)**

**Import safety**: No top-level side effects. Imports are `__future__`, `typing`, `numpy`, `pandas` only. No filesystem calls at module level.

**Constants**:
- `RUNNER_ID = "M2_STRUCTURAL_RUNNER_BO01_MR02_V1"` ✓ correct
- `ALLOWED_STRATEGY_IDS = {"BO01", "MR02"}` ✓ correct, only two strategies
- `FORBIDDEN_PERFORMANCE_TERMS` ✓ negative-only list, not used in active logic

**`validate_frame_for_m2`**:
- Rejects None/empty frame ✓
- Requires `pd.DatetimeIndex` ✓ (Note: timestamp-as-column pattern not supported — minor limitation, acceptable since future M2 runner can pre-convert)
- Requires timezone-aware index ✓
- Blocks 2025 and 2026 years via `raise ValueError` ✓
- Checks partition/split/dataset_split/data_split columns for "val" and "hold" substrings ✓
- Requires open/high/low/close ✓
- Checks approximate M5 cadence (1–10 min median) ✓
- Checks OHLC NaN ✓
- No performance calculations anywhere in this function ✓

**`run_structural_counts`**:
- Accepts only BO01/MR02 via `ALLOWED_STRATEGY_IDS` check ✓
- Calls `strategy_cls.default_params()` safely when params is None ✓
- Per-bar forbidden date check (2025/2026) with `fail_closed_count` ✓
- Per-bar partition/split check for "val"/"hold" with `fail_closed_count` ✓
- OHLC finite check per bar with `fail_closed_count` ✓
- Wraps `strategy_cls.signal()` in try/except; exceptions increment `exception_count` ✓
- No trade list generated ✓
- No PnL, Sharpe, winrate, drawdown, or expectancy calculated ✓
- No file I/O ✓
- No mutation of `params` dict ✓
- **W-01**: `valid_signal_count` is incremented at line 181 before the contract check at lines 185–190. If the contract check raises (e.g., a strategy returns a dict without `"signal"` key), the bar is counted in `valid_signal_count` and then also in `exception_count`/`fail_closed_count`. The final `contract_valid_count` correctly reflects only fully-validated signals. For structural counting purposes this is a minor ambiguity in semantics, not a correctness blocker.

**`run_m2_structural_evaluation`**:
- Window filtering via `.loc[start_ts:end_ts].copy()` — uses `.copy()`, no frame mutation ✓
- Window not extended by results ✓
- Calls `validate_frame_for_m2` on the sliced frame ✓
- No file read or write ✓
- No data loading ✓
- Returns in-memory dict summary ✓

---

## 7. Contract Compatibility Audit

**Result: PASS — compatible with BO01 and MR02 signal contracts**

BO01 signal function (`BO01Strategy.py`, line 177) signature: `signal(frame, i, params) -> dict | None`
- Returns dict with keys: `signal` (1 or -1), `direction` ("long"/"short"), `stop_mode`, `stop_price`, `target_mode`, `target_rr`, `break_even_at_r`, `trailing_atr`, `session_name`
- Runner checks `"signal"` and `"direction"` keys ✓ — these are present in all BO01 signal outputs

MR02 signal function (`MR02Strategy.py`, line 197) signature: `signal(frame, i, params) -> dict | None`
- Returns dict with same `_build_signal` structure: `signal`, `direction`, `stop_mode`, `stop_price`, `target_mode`, `target_rr`, `break_even_at_r`, `trailing_atr`, `session_name`
- Runner checks `"signal"` and `"direction"` keys ✓ — these are present in all MR02 signal outputs

Both strategies return `None` for no-signal bars ✓ — runner handles `None` correctly.

---

## 8. Test Audit

**Result: PASS_TESTS_SYNTHETIC_SAFE (with W-02)**

**Contract tests** (`test_m2_structural_runner_contract.py`):
- All frames constructed in-memory using `pd.date_range` + `pd.DataFrame` ✓
- No CSV files read ✓
- No data vault path referenced ✓
- 2025/2026 used only in negative test cases (fail-closed verification) ✓
- validation/holdout used only in negative test cases ✓
- No persistent outputs created ✓
- No performance metrics checked ✓
- FakeStrategyBO01 correctly exercises: None path, exception path, valid signal path ✓

**Safety tests** (`test_m2_structural_runner_safety.py`):
- FakeStrategyMR02 always returns None — correct for safety isolation ✓
- `test_no_filesystem_writing` uses `os.listdir(".")` before/after to verify no new files — acceptable technique ✓
- `test_no_frame_mutation` uses `pd.testing.assert_frame_equal` — correct ✓
- `test_runner_does_not_mutate_params` verifies dict identity after run ✓
- **W-02**: `test_no_forbidden_active_logic_terms` uses `open(runner_path)` to read the runner source, then only checks that `FORBIDDEN_PERFORMANCE_TERMS` exists and contains three specific keys. This is a shallow check — it does not verify active logic paths are free of these terms. Acceptable as supplementary evidence alongside the static scan, but insufficient as a standalone contract guard.

**Coverage gaps (non-blocking)**:
- No test for strategy_id fallback path (when `strategy_cls` has no `ID` attribute)
- No test for window boundary edge cases in `run_m2_structural_evaluation`
- No test for mixed train+validation partition frames (only pure holdout tested)

---

## 9. Test Execution Results

All tests executed on audit branch with `PYTHONPATH=03_RESEARCH_LAB`. No real data loaded.

| Suite | Tests | Result |
|---|---|---|
| test_m2_structural_runner_contract | 10 | OK |
| test_m2_structural_runner_safety | 4 | OK |
| test_strategy_contract_bo01 | 11 | OK |
| test_strategy_contract_mr02 | 12 | OK |
| **TOTAL** | **37** | **OK — 0 failures** |

**Classification: PASS_TEST_EXECUTION_SYNTHETIC_ONLY**

---

## 10. Static Safety Scan

**Classification: PASS — no blockers**

| Pattern | File | Context | Classification |
|---|---|---|---|
| `read_csv` | none | — | PASS — absent |
| `to_csv` | none | — | PASS — absent |
| `data_vault` / `05_MARKET_DATA_VAULT` | none | — | PASS — absent |
| `EURUSD_M5` hardcoded | none | — | PASS — absent |
| `open(` | safety test L30 | reads runner's own `__file__` path for self-inspection | STATIC_READ_SELF_OK |
| performance terms (pnl, winrate, etc.) | runner L11–34 | `FORBIDDEN_PERFORMANCE_TERMS` tuple — negative declaration only | NEGATIVE_DECLARATION_OK |
| `2025`/`2026` | runner L54, L134 | `raise ValueError` guards — negative enforcement | TEST_NEGATIVE_CASE_OK |
| `validation`/`holdout` | runner L57–62, L143–144 | negative enforcement guards | TEST_NEGATIVE_CASE_OK |
| `equity_curve.csv` / `trades.csv` in `git ls-files` | `07_BACKUPS/legacy_archive_2026/...` | pre-existing legacy archive, untouched, out of scope | GOVERNANCE_TERM_OK — pre-existing |

**Blockers: 0**

---

## 11. Git / Output Security Audit

**Result: PASS_GIT_OUTPUT_SECURITY**

- No local outputs committed in this branch delta ✓
- No market data committed in this branch delta ✓
- No data vault modified ✓
- No secrets or credentials found ✓
- No non-whitelisted files staged in this branch ✓
- Pre-existing legacy `trades.csv`/`equity_curve.csv` in `07_BACKUPS` noted as **WARN_PREEXISTING_DIRTY_UNTOUCHED** — pre-dates this audit, not in scope, not touched

---

## 12. Findings Table

| id | severity | category | finding | evidence | implication | required_action |
|---|---|---|---|---|---|---|
| F-01 | WARN | runner_semantics | `valid_signal_count` incremented before contract check | runner L181 vs L185–193 | Minor semantic ambiguity: a signal that fails contract check is counted in `valid_signal_count` AND `exception_count`. `contract_valid_count` is correct. | No blocker. Can be patched in a future minor fix by moving the increment to inside the contract-valid block. |
| F-02 | WARN | test_coverage | Safety test `test_no_forbidden_active_logic_terms` performs shallow self-read check | test L29–38 | Confirms list existence but does not verify active logic paths | No blocker. Static scan compensates. Acceptable as-is for this phase. |
| F-03 | INFO | test_coverage | No test for strategy_id fallback when `ID` attribute absent | test files | Unlikely to affect BO01/MR02 which both define `ID` | No action required at this stage |
| F-04 | INFO | preexisting | Legacy `trades.csv`/`equity_curve.csv` in `07_BACKUPS` visible in `git ls-files` | `git ls-files` output | Pre-existing, unrelated to this phase | No action — out of scope for this audit |

---

## 13. Decision

**`M2_STRUCTURAL_RUNNER_AUDIT_PASS_WITH_WARNINGS`**

The runner is structurally sound and safe for its declared purpose:
- Counting structural signal activity on train-only data frames
- Enforcing fail-closed on 2025/2026, validation, and holdout partitions
- Accepting only BO01 and MR02 strategies
- Producing zero filesystem side effects

The two warnings (W-01, W-02) are minor and do not compromise the runner's safety or its structural counting accuracy for the intended M2 Conservative evaluation.

**This audit does NOT**:
- Execute M2
- Prove edge or profitability of BO01 or MR02
- Authorize use of validation or holdout data
- Authorize use of 2025 or 2026 data
- Replace a formal backtest or training evaluation
- Constitute a certification of strategy readiness

**Owner decision is required** before any M2 Conservative structural execution is retried.

---

## 14. Allowed Next Step

**A) Owner decision whether to reattempt M2 Conservative structural execution.**

If the owner approves, the next phase would use the audited runner (`m2_structural_runner`) to count structural signals on train-only data (2015–2024), with no performance metrics computed.

Alternatively:
- **B) Patch W-01** (move `valid_signal_count` increment to after contract check) — optional minor improvement
- **C) Block** — owner does not proceed

---

## 15. Forbidden Next Steps

- No immediate M2 execution from this audit alone
- No backtest
- No formal training
- No validation data
- No holdout data
- No 2025 or 2026 data
- No optimization or sweep
- No demo, real, or FTMO
- No declaration of edge or robustness
