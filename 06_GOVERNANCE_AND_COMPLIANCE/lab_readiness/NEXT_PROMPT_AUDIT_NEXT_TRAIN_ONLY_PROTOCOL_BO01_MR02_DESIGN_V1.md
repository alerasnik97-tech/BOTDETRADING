# NEXT PROMPT — AUDIT NEXT TRAIN-ONLY PROTOCOL BO01/MR02 DESIGN V1

Act as a **READ-ONLY EXTERNAL QUANTITATIVE AUDITOR** of the Trading BOT project. Your single mission is to audit the design of the proposed `M2_TRAIN_ONLY_LIMITED_STRUCTURAL_EVALUATION_PROTOCOL` before any execution can be authorized.

---

## 1. Scope and Target files
You must inspect only the following markdown files:
1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_AUDIT_PUBLICATION_VERIFICATION_REPORT_V1.md`
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_TRAIN_ONLY_PROTOCOL_BO01_MR02_DESIGN_V1.md`
3. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_TRAIN_ONLY_PROTOCOL_BO01_MR02_DESIGN_REPORT_V1.md`

---

## 2. Audit Verification Points
You must verify the following design parameters:
- **M1 Audit Publication Parity:** Check that the M1 audit commit exists on both the local and remote repositories at HEAD SHA `10f2caf8507c135c59a66505b3ee36d19ed301ba`.
- **M2 Data Policy Integrity:** Ensure that M2 is strictly limited to prepared train-only data (`EURUSD_M5.csv`) for the range 2015-01-01 to 2024-12-31. Prohibit any 2025/2026, validation, or holdout data access.
- **M2 Metrics Policy Cleanliness:** Verify that the protocol explicitly prohibits calculating performance metrics (PnL, Drawdown, Win Rate, Profit Factor, Sharpe). Confirm that it only allows structural metrics (row_count, calls, signals, session distribution).
- **M2 Output Policy Insulated:** Check that all local output file definitions are restricted to gitignored directories under `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/`.
- **Runner Policy Preservation:** Ensure that no runner or core code modifications are designed or allowed without external audit.
- **Abort Conditions robust:** Confirm that standard quantitative risk constraints are fully documented.

---

## 3. Strict Safety Boundaries
- **NO execution is authorized by this audit.**
- **NO data loading is permitted.**
- **NO code or test modifications are allowed.**
- **NO PnL, Win Rate, Sharpe, or Drawdown calculations are allowed.**
