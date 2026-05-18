# NEXT PROMPT — AUDIT M2 CONSERVATIVE EXECUTION PROMPT DRAFT V1

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
Verify the integrity, compliance, and language of the drafted M2 Conservative Train-Only Structural Evaluation prompt and the sanity corrections applied.

---

## 2. Verification Checklist

### 2.1 SHA and Lineage verification
- Confirm that the base branch is `audit/m2-design-audit-sane-review-v1-20260518`.
- Confirm that the draft branch is `research/draft-m2-conservative-structural-execution-prompt-v1-20260518`.
- Verify the physical local HEAD SHA is `e83f5a5a53268e8095bb8f22a79d7fa0934362c4`.
- Verify that the incorrect handoff SHA `e83f5c5ae04d5570220fb079b940989f64bfbb8e` is explicitly declared as incorrect and clarified in the files.
- Verify the M2 Design real SHA is `aba333a0379a4f733afa39180462eddd68c02656` and the M1 Audit verified SHA is `10f2caf8507c135c59a66505b3ee36d19ed301ba`.

### 2.2 Language Neutralization
- Confirm that absolute, superlative, or hyper-optimistic claims (such as "fully sane", "validated", "completely untouched", "perfectly", "successfully", "100%", "sealed", "certified") are eliminated from revised and newly created files.
- Verify that all remaining occurrences of "strictly" are only used for negative prohibitions.

### 2.3 Owner Prompt Ambiguity Fix
- Confirm that in `NEXT_PROMPT_OWNER_DECIDES_M2_AFTER_SANITY_AUDIT_V1.md`, Option A has been corrected from "Draft and Execute" to "Draft M2 Conservative execution prompt for later audit".
- Verify that it clarifies that no immediate execution is enabled under that phase.

### 2.4 Future M2 Execution Prompt Compliance
Check `NEXT_PROMPT_EXECUTE_M2_CONSERVATIVE_TRAIN_ONLY_BO01_MR02_V1.md` for:
- **Activation Gate:** The exact required future autonomous phrase is present:
  “APRUEBO EJECUTAR M2 CONSERVATIVE TRAIN-ONLY STRUCTURAL EVALUATION BO01/MR02, SOLO MÉTRICAS ESTRUCTURALES, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN BACKTEST, SIN TRAIN FORMAL Y SIN OPTIMIZATION/SWEEP.”
- **Data Policy:** The target is strictly `EURUSD_PREPARED_TRAIN_2015_2024_M5` within the window `2015-01-01` to `2015-03-31` (UTC). Hard blocks exist for validation, holdout, 2025, and 2026 partitions.
- **Runner Policy:** Checks for audited runner pre-existence before execution; prohibits runner or core code modifications. Prohibits non-gitignored temporary scripts.
- **Allowed Metrics:** Prohibits performance/profitability metrics (PnL, winrate, drawdown, Profit Factor, Sharpe/Sortino, expectancy). Permits only structural/robustness/fail-closed counts.
- **Output Policy:** Designated gitignored local path only, with clear allowed and forbidden outputs.
- **Abort Conditions:** Clear conditions to immediately halt if any violation, worktree instability, or code mutation occurs.
- **No Owner-less Path:** Prohibits executing without the exact standalone phrase.

---

## 3. Allowed Methods
The auditor is permitted to use **ONLY** read-only text commands:
- Git inspection commands (`git status`, `git branch`, `git log`, `git diff`).
- Text search commands (`rg` or native PowerShell search commands).
- Reading markdown files using file viewers.

---

## 4. Final Audit Decision
The auditor must report a final safety status:
- **STATUS = PASS:** If all checks comply, lineage is tracked, language is neutral, and the execution draft is fully compliant.
- **STATUS = BLOCKER:** If any python script execution is allowed, any performance metrics are permitted, or any execution is possible without the exact activation gate phrase.
