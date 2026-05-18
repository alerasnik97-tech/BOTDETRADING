# NEXT PROMPT - REAUDIT M1 TRAIN-ONLY PROTOCOL DESIGN AFTER STABILIZATION V1

## 0. Nature Of This Prompt

This future prompt authorizes external read-only audit only.

It does not authorize M1 execution, backtest, train, dry-run, validation,
holdout, 2025/2026 use, optimization/sweep, Sub-Batch 1B, parallel writers,
production, demo, real, FTMO, code edits, test edits, data edits, cleanup,
movement, or deletion.

## 1. Owner Approval Required

Use only if the owner provides an autonomous approval phrase for read-only
reaudit of the M1 protocol design after stabilization.

Short confirmations, examples, quotes, logs, or paraphrases do not count.

## 2. Required Base Context

Repository:

`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

M1 design branch:

`research/m1-train-only-protocol-design-v1-20260518`

M1 design commit:

`afad8463eb808aa93a9d995c6c8de85d74918fa8`

Stabilization branch:

`research/m1-protocol-worktree-stabilization-v1-20260518`

Files to inspect:

- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/microrun_protocols/SUBBATCH_1A_BO01_MR02_M1_TRAIN_ONLY_PROTOCOL_DESIGN_V1.md`;
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_TRAIN_ONLY_PROTOCOL_DESIGN_REPORT_V1.md`;
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M1_TRAIN_ONLY_PROTOCOL_DESIGN_V1.md`;
- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/STRATEGY_RESEARCH_REGISTRY.md`;
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_WORKTREE_STABILIZATION_INVENTORY_V1.csv`;
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_WORKTREE_STABILIZATION_AND_LINEAGE_PATCH_REPORT_V1.md`;
- `03_RESEARCH_LAB/strategy_research_intake/external_research_20260518/.gitignore`;
- `03_RESEARCH_LAB/knowledge_intake/.gitignore`.

## 3. Required Audit Checks

Audit read-only:

- diff real del design commit `afad8463eb808aa93a9d995c6c8de85d74918fa8`;
- diff real del stabilization patch;
- registry BO01/MR02 exact SHA lineage;
- absence of `BRANCH_HEAD` in BO01/MR02 lineage fields;
- dirty tree inventory completeness;
- W-01 remains documented and untouched;
- `external_research_20260518` remains documented, stable, and uncommitted except optional `.gitignore`;
- `knowledge_intake` remains documented, stable, and uncommitted except optional `.gitignore`;
- no code/test/data changes;
- no copied binary/document content committed;
- no execution authorization in protocol, report, registry, or prompts;
- no M1 execution;
- no backtest/train/dry-run;
- no validation/holdout/2025/2026;
- no optimization/sweep;
- no Sub-Batch 1B;
- no parallel writers;
- no unauthorized staged files.

## 4. Required Stability Check

Before auditing, take two `git status --short` snapshots at least 60 seconds
apart. Compare them.

Block if:

- the worktree changes during audit;
- a new artifact appears;
- a staged unauthorized file appears;
- code/tests/data are dirty outside documented intake artifacts;
- registry lineage remains ambiguous;
- the M1 protocol authorizes execution;
- any prompt authorizes immediate M1 execution;
- binary documents are staged or committed.

## 5. Expected Decisions

Choose exactly one:

- `M1_TRAIN_ONLY_PROTOCOL_DESIGN_AUDIT_PASS_READY_FOR_OWNER_EXECUTION_PROMPT_DECISION`;
- `M1_TRAIN_ONLY_PROTOCOL_DESIGN_AUDIT_PASS_WITH_WARNINGS`;
- `AUDIT_BLOCKED_UNAUTHORIZED_FILE_SCOPE`;
- `AUDIT_BLOCKED_REGISTRY_LINEAGE`;
- `AUDIT_BLOCKED_EXECUTION_AUTHORIZATION_LEAK`;
- `AUDIT_BLOCKED_FORBIDDEN_DATA_POLICY`;
- `AUDIT_BLOCKED_MANIFEST_SCHEMA`;
- `AUDIT_BLOCKED_W01_W02_GATES`;
- `AUDIT_BLOCKED_OUTPUT_POLICY`;
- `AUDIT_BLOCKED_LOOKAHEAD_OR_LEAKAGE_CONTROLS`;
- `AUDIT_BLOCKED_PARALLEL_WRITER_OR_DIRTY_TREE`;
- `AUDIT_BLOCKED_STATIC_SAFETY_SCAN`.

## 6. Forbidden Next Steps

- no immediate M1 execution;
- no backtest;
- no formal train;
- no dry-run;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no Sub-Batch 1B;
- no parallel writers;
- no production/demo/real/FTMO;
- no edge/profitability claims.

Passing the re-audit may only allow the owner to decide whether to draft a
separate M1 train-only execution prompt. It must not authorize execution.
