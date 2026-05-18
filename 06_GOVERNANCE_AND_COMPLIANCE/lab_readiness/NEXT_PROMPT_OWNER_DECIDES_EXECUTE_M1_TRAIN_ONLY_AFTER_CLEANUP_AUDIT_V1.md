# NEXT PROMPT — OWNER DECIDES EXECUTE M1 TRAIN-ONLY AFTER CLEANUP AUDIT V1

## 0. Nature Of This Document
This is a governance-read-only template designed to present options to the owner regarding whether to authorize execution of the audited M1 train-only prompt.
- No execution is authorized now.
- The exact autonomous activation phrase is still required to proceed.
- No backtest, train, validation, holdout, or sweep is allowed.

---

## 1. Status
**`M1_TRAIN_ONLY_EXECUTION_PROMPT_CLEANUP_AUDIT_PASS_WITH_WARNINGS`**

---

## 2. Options For The Owner
The owner must select exactly one of the following options in their next prompt:

### Option A: Approve execution of audited M1 train-only prompt
Selecting this option authorizes the next agent to execute the audited M1 train-only prompt (M1A metadata preflight and M1B tiny controlled execution slice).
The owner must provide the exact, autonomous activation phrase at the top level of their prompt:

`“APRUEBO EJECUTAR M1 TRAIN-ONLY BO01/MR02, SOLO M1A METADATA PREFLIGHT Y M1B TINY CONTROLLED EXECUTION, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN BACKTEST, SIN TRAIN FORMAL Y SIN OPTIMIZATION/SWEEP.”`

*Implication:* The next agent will execute the M1A preflight, load the tiny contiguous train-only M1B slice, check timezone and fail-closed pathways, generate the output logs, and write the final manifest and execution reports. No backtest or parameter sweeps will occur.

### Option B: Request minor patch
The owner may request modifications to the audit report, findings table, or gitignore guards before proceeding.

### Option C: Do not advance
The candidates will remain at their current status (`M1_TRAIN_ONLY_EXECUTION_PROMPT_DRAFT_PENDING_AUDIT` updated in registry) without executing anything.

---

## 3. Strict Restrictions
- **No execution now:** Selecting Option A does NOT authorize any strategy execution or data loading until the next prompt is processed.
- **No backtest:** Backtesting remains prohibited.
- **No train:** Running standard training scripts remains blocked.
- **No validation/holdout/2025/2026:** All validation and holdout partitions remain sealed.
- **No optimization/sweep:** Parameter tuning or grids remain prohibited.
- **No Sub-Batch 1B:** Restrained to design sub-phase.
- **No parallel writers.**
- **No production/demo/real/FTMO accounts.**
