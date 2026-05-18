# NEXT PROMPT — OWNER DECIDES EXECUTE M0 SYNTHETIC-ONLY V1

## 0. Nature Of This Document
This is a governance-read-only template designed to present options to the owner regarding the potential execution of the already-audited M0 synthetic-only execution prompt for BO01 and MR02.
- No execution is authorized now.
- No real data is exposed.
- No data vault access is permitted.
- No dry-run, backtest, train, validation, holdout, or sweep is allowed.

---

## 1. Status
**`M0_SYNTHETIC_EXECUTION_PROMPT_AUDIT_PASS_READY_FOR_OWNER_USE_DECISION`**

---

## 2. Options For The Owner
The owner must explicitly select one of the following options in their next prompt:

### Option A: Approve execution of already-audited M0 synthetic-only prompt
If selecting this option, the owner must provide the exact activation phrase in their prompt:
“APRUEBO EJECUTAR M0 SYNTHETIC-ONLY MICRORUN BO01/MR02, SIN DATOS REALES, SIN DATA VAULT, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

*Implication:* This will authorize the next agent to switch to the execution branch, create the tiny in-memory M5 bar fixtures, run the signal tests, write the synthetic microrun report under `local_outputs_do_not_commit/`, and abort if any real files, data vault, or backtest runners are accessed.

### Option B: Request minor patch before execution
The owner may request modifications to the execution prompt structure or test cases before proceeding.

### Option C: Do not advance
The candidates will remain at their current status (`M0_SYNTHETIC_EXECUTION_PROMPT_DRAFT_PENDING_AUDIT` promoted in registry) without executing any signal calls.

---

## 3. Strict Restrictions
- **No immediate execution now:** Selecting Option A still requires the next agent to receive the exact activation phrase in the next prompt turn.
- **No real data:** Any access to real historical or tick datasets remains completely prohibited.
- **No data vault:** Any connection to `05_MARKET_DATA_VAULT` is locked.
- **No validation/holdout/2025/2026:** All validation and holdout sets remain sealed.
- **No backtest/train/sweep:** Any dynamic parameter tuning or backtesting is prohibited.
- **No production/demo/real/FTMO accounts.**
