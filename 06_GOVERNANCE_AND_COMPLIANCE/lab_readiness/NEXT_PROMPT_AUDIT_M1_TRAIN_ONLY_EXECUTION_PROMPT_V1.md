# NEXT PROMPT — AUDIT M1 TRAIN-ONLY EXECUTION PROMPT V1

## 0. Nature Of This Document
This is a future external read-only audit prompt. It authorizes no dynamic execution, no strategy calls, and no data loading.
Its only purpose is to perform a read-only verification of the drafted M1 execution prompt and its compliance with the laboratory's safety guidelines.

---

## 1. Required Scope
The auditing agent must inspect the following and verify compliance:
- **Diff scope:** Verify that the git diff contains exclusively markdown files and whitelisted registry updates. No code, tests, or market data must be touched.
- **Exact SHA Lineage:** Verify that all strategy candidates (BO01 and MR02) are mapped in the registry to the exact design commit SHA `1f69e2b0c5a49a0b97fe4ff2ac317e0547951ad8` and that NO `BRANCH_HEAD` placeholders remain.
- **No Wording Inflation:** Verify that no absolute or hyper-positive qualitative qualifiers (e.g., "100%", "fully", "successfully", "certified", "locked", "strictly and absolutely", "perfectly", "sealed") are used to describe current or future execution states.
- **Runner Policy:** Verify that the runner policy strictly prohibits the creation of new runner files inside production/incubation directories or the modification of core code. If a runner is used, it must be verified as audited, or the execution aborts with `BLOCKED_M1_RUNNER_NOT_AUDITED_OR_NOT_FOUND`.
- **No Code Creation:** Verify that the prompt authorizes no dynamic code creation during future execution, unless it is a temporary script in the ignored local directory.
- **No Data Loading during Audit:** Verify that no data is loaded or processed during this read-only audit phase.
- **W-01 / W-02:** Verify that pre-existing backlog W-01 and W-02 remain active gates and are not affected.
- **Broad Gitignore Warning:** Verify that the broad `.gitignore` warnings are documented and understood.
- **No Owner-less Execution Path:** Confirm that the drafted prompt (`NEXT_PROMPT_EXECUTE_M1_TRAIN_ONLY_BO01_MR02_V1.md`) requires the exact autonomous owner approval phrase before any action.

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
