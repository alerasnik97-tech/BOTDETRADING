# NEXT PROMPT — AUDIT M1 TRAIN-ONLY EXECUTION PROMPT V1

## 0. Nature Of This Document
This is a future external read-only audit prompt. It authorizes no dynamic execution, no strategy calls, and no data loading.
Its only purpose is to perform a rigorous read-only verification of the drafted M1 execution prompt and its compliance with the laboratory's safety guidelines.

---

## 1. Required Scope
Audit read-only:
- **Diff scope:** Verify that the git diff contains exclusively markdown files and whitelisted registry updates. No code, tests, or market data must be touched.
- **Activation Gate:** Confirm that the drafted prompt (`NEXT_PROMPT_EXECUTE_M1_TRAIN_ONLY_BO01_MR02_V1.md`) requires the exact autonomous owner approval phrase.
- **Sub-phase definitions:** Confirm that M1A is strictly restricted to metadata preflight and M1B to a tiny contiguous controlled slice.
- **Data Policy:** Confirm that the target dataset is `EURUSD_PREPARED_TRAIN_2015_2024_M5` and that any 2025/2026 data immediately triggers an execution abort.
- **Strict Prohibitions:** Verify that standard backtests, training runners, validation/holdout sets, parallel writers, and optimization sweeps are strictly banned.
- **Output Policy:** Verify that allowed outputs are directed only to ignored directories under `local_outputs_do_not_commit/` and that standard files (`trades.csv`, `equity_curve.csv`, ZIPs) are prohibited.
- **Manifest Schema:** Verify that a complete, hardened manifest schema with individual file hashes and true/false verification flags is declared.
- **W-01 / W-02 Gates:** Verify that pre-existing backlog W-01 and W-02 remain active gates and are not affected.

---

## 2. Files To Inspect
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_EXECUTE_M1_TRAIN_ONLY_BO01_MR02_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_TRAIN_ONLY_EXECUTION_PROMPT_DRAFT_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M1_TRAIN_ONLY_EXECUTION_PROMPT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/STRATEGY_RESEARCH_REGISTRY.md`

---

## 3. Prohibited Actions During Audit
- **NO Strategy calls or execution.**
- **NO Price calculations or data loading.**
- **NO Backtesting or training.**
- **NO Optimization or parameter sweeps.**
- **NO Modification to `BO01Strategy.py`, `MR02Strategy.py`, tests, or engine.**
- **NO Git force-push, merge, or rebase.**
- **NO Staging of unauthorized files.**

---

## 4. Expected Audit Decisions
The auditing agent must select exactly one of the following decisions:
- `M1_TRAIN_ONLY_EXECUTION_PROMPT_AUDIT_PASS_READY_FOR_OWNER_USE_DECISION`
- `M1_TRAIN_ONLY_EXECUTION_PROMPT_AUDIT_PASS_WITH_WARNINGS`
- `AUDIT_BLOCKED_UNAUTHORIZED_FILE_SCOPE`
- `AUDIT_BLOCKED_EXECUTION_AUTHORIZATION_LEAK`
- `AUDIT_BLOCKED_FORBIDDEN_DATA_POLICY`
- `AUDIT_BLOCKED_MANIFEST_SCHEMA`
- `AUDIT_BLOCKED_W01_W02_GATES`
- `AUDIT_BLOCKED_OUTPUT_POLICY`
- `AUDIT_BLOCKED_LOOKAHEAD_OR_LEAKAGE_CONTROLS`
- `AUDIT_BLOCKED_REGISTRY_SCOPE`
- `AUDIT_BLOCKED_STATIC_SAFETY_SCAN`

---

## 5. Forbidden Next Steps After Audit
- **NO immediate M1 execution.**
- **NO backtest.**
- **NO formal train.**
- **NO validation.**
- **NO holdout.**
- **NO 2025/2026.**
- **NO optimization/sweep.**
- **NO Sub-Batch 1B.**
- **NO parallel writers.**
- **NO production/demo/real/FTMO.**
- **NO edge/profitability claims.**
