# NEXT PROMPT — AUDIT M2 CONSERVATIVE DRAFT AFTER LINEAGE PATCH V1

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
Verify the correct lineage tracing, physical SHA verification, and administrative bounds of the drafted M2 Conservative Train-Only Structural Evaluation prompt.

---

## 2. Verification Checklist

### 2.1 SHA and Lineage Verification
- Verify that **no** occurrences of the incorrect SHA prefix `e83f5a5a` remain in the draft files.
- Confirm that the physical base sanity HEAD SHA is documented as `e83f5c5a53268e8095bb8f22a79d7fa0934362c4`.
- Confirm that the incorrect handoff SHA `e83f5c5ae04d5570220fb079b940989f64bfbb8e` remains only as a historical reference to the prior turn's bad data.
- Confirm the merge lineage: base is `audit/m2-design-audit-sane-review-v1-20260518`, draft branch is `research/draft-m2-conservative-structural-execution-prompt-v1-20260518`, and current patch branch is `research/fix-m2-conservative-draft-lineage-v1-20260518`.
- Verify the M2 Design real SHA is `aba333a0379a4f733afa39180462eddd68c02656` and the M1 Audit verified SHA is `10f2caf8507c135c59a66505b3ee36d19ed301ba`.

### 2.2 Future M2 Execution Prompt Compliance
Check `NEXT_PROMPT_EXECUTE_M2_CONSERVATIVE_TRAIN_ONLY_BO01_MR02_V1.md` for:
- **Activation Gate:** The exact required future autonomous phrase is present:
  “APRUEBO EJECUTAR M2 CONSERVATIVE TRAIN-ONLY STRUCTURAL EVALUATION BO01/MR02, SOLO MÉTRICAS ESTRUCTURALES, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN BACKTEST, SIN TRAIN FORMAL Y SIN OPTIMIZATION/SWEEP.”
- **Data Policy:** Target dataset is strictly `EURUSD_PREPARED_TRAIN_2015_2024_M5` within the window `2015-01-01` to `2015-03-31` (UTC). Validation, holdout, 2025, and 2026 partitions are blocked.
- **Runner Policy:** Checks for audited runner pre-existence before execution; prohibits runner or core code modifications. Prohibits non-gitignored temporary scripts.
- **Allowed Metrics:** Prohibits performance/profitability metrics (PnL, winrate, drawdown, Profit Factor, Sharpe/Sortino, expectancy). Permits only structural/robustness/fail-closed counts.
- **Output Policy:** Designated gitignored local path only, with clear allowed and forbidden outputs.
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
- **STATUS = PASS:** If all checks comply, lineage is corrected to match physical HEAD, no incorrect SHA remains, language is neutral, and the execution draft is compliant with the reviewed restrictions.
- **STATUS = BLOCKER:** If any python script execution is allowed, any performance metrics are permitted, or any execution is possible without the exact activation gate phrase.
