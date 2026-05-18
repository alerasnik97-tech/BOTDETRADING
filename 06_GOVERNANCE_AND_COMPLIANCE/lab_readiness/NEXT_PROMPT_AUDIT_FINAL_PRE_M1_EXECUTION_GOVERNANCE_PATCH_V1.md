# NEXT PROMPT — AUDIT FINAL PRE-M1 EXECUTION GOVERNANCE PATCH V1

## 0. Nature Of This Document
This is a future external read-only audit prompt. It authorizes no dynamic execution, no strategy calls, and no data loading.
Its only purpose is to perform a read-only verification of the final pre-M1 execution governance patch and its compliance with the laboratory's safety guidelines.

---

## 1. Required Scope
The auditing agent must inspect the following and verify compliance:
- **Diff scope:** Verify that the git diff contains exclusively markdown files and whitelisted registry updates. No code, tests, or market data must be touched.
- **Exact SHA Lineage:** Verify that all strategy candidates (BO01 and MR02) are mapped in `STRATEGY_RESEARCH_REGISTRY.md` to the exact cleanup commit SHA `7272b8513ab4cf78cbd94ecf0f71e2a41a42658b` and that NO `BRANCH_HEAD` placeholders remain.
- **No Language Inflation:** Verify that no absolute qualifiers (e.g., successfully, certified, perfect, airtight, fully, sealed, locked, 100%) are used to describe current or future execution states.
- **Git Reset Blocker Warning:** Verify that the findings table in `M1_TRAIN_ONLY_EXECUTION_PROMPT_CLEANUP_EXTERNAL_AUDIT_V1.md` documents the historical incident of the prohibited `git reset --hard` (F-06), and that no code/tests/data changes occurred.
- **No Code/Test/Data changes:** Confirm that no code, tests, or raw dataset files were modified.
- **No Data Loading during Audit:** Verify that no data is loaded or processed during this read-only audit phase.
- **W-01 / W-02:** Verify that pre-existing backlog W-01 and W-02 remain active gates and are not affected.
- **No Reset/Rebase/Clean/Stash:** Verify that no reset, rebase, clean, or stash commands were used in this phase.
- **Staged files only:** Verify that staged files are strictly authorized markdowns only.

---

## 2. Files To Inspect
- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/STRATEGY_RESEARCH_REGISTRY.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_TRAIN_ONLY_EXECUTION_PROMPT_CLEANUP_EXTERNAL_AUDIT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_OWNER_DECIDES_EXECUTE_M1_TRAIN_ONLY_AFTER_CLEANUP_AUDIT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FINAL_PRE_M1_EXECUTION_GOVERNANCE_PATCH_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_FINAL_PRE_M1_EXECUTION_GOVERNANCE_PATCH_V1.md`

---

## 3. Prohibited Actions During Audit
- **NO Strategy calls or execution.**
- **NO Price calculations or data loading.**
- **NO Backtesting or training.**
- **NO Optimization or parameter sweeps.**
- **NO Modification to `BO01Strategy.py`, `MR02Strategy.py`, tests, or engine.**
- **NO Git reset --hard, clean, stash, rebase, or force-push.**
- **NO Git add dot.**

---

## 4. Expected Audit Decisions
The auditing agent must select exactly one of the following decisions:
- `FINAL_PRE_M1_GOVERNANCE_PATCH_PASS_READY_FOR_OWNER_USE_DECISION`
- `FINAL_PRE_M1_GOVERNANCE_PATCH_PASS_WITH_WARNINGS`
- `AUDIT_BLOCKED_UNAUTHORIZED_FILE_SCOPE`
- `AUDIT_BLOCKED_REGISTRY_LINEAGE_GAP`
- `AUDIT_BLOCKED_LOOKAHEAD_OR_LEAKAGE_CONTROLS`
- `AUDIT_BLOCKED_W01_W02_GATES`
- `AUDIT_BLOCKED_OUTPUT_POLICY`
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
