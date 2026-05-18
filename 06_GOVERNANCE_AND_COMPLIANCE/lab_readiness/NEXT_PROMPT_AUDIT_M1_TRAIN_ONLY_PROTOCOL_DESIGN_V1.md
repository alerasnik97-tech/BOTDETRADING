# NEXT PROMPT - AUDIT M1 TRAIN-ONLY PROTOCOL DESIGN V1

## 0. Nature Of This Prompt

This is a future external read-only audit prompt. It authorizes no execution and
no file modification except an audit markdown if separately approved by the
owner in that future phase.

It must audit the M1 train-only protocol design for BO01/MR02 only.

## 1. Required Scope

Audit read-only:

- real diff;
- protocol design;
- design report;
- registry update for BO01/MR02 only;
- data policy;
- manifest schema;
- W-01/W-02 gates;
- output policy;
- anti-lookahead controls;
- data-leakage controls;
- metrics policy;
- no execution authorization;
- no code/test/data changes;
- no validation/holdout;
- no 2025/2026 data;
- no optimization/sweep;
- no Sub-Batch 1B;
- no parallel writers.

## 2. Files To Inspect

- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/microrun_protocols/SUBBATCH_1A_BO01_MR02_M1_TRAIN_ONLY_PROTOCOL_DESIGN_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_TRAIN_ONLY_PROTOCOL_DESIGN_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M1_TRAIN_ONLY_PROTOCOL_DESIGN_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/STRATEGY_RESEARCH_REGISTRY.md`
- the base audit artifact:
  `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_MICRORUN_EXECUTION_EXTERNAL_AUDIT_V1.md`

## 3. Prohibited Actions

- no code modification;
- no test modification;
- no data modification;
- no M1 execution;
- no M0 execution;
- no backtest;
- no train;
- no dry-run;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no grid search;
- no walk-forward;
- no Sub-Batch 1B;
- no parallel writers;
- no production/demo/real/FTMO;
- no `git add .`;
- no force push;
- no destructive Git commands.

## 4. Audit Checks

The audit must verify:

- branch is not `main`;
- diff includes only authorized markdown files;
- no code/test/data file changed;
- registry update, if present, touches BO01/MR02 only;
- protocol status is design-only and pending external audit;
- M1A/M1B/M1C are design sub-phases only;
- data policy limits future data to EURUSD train-only, maximum
  `2015-01-01` through `2024-12-31`;
- no validation, holdout, 2025, or 2026 is authorized;
- manifest schema fixes M0 provenance warning;
- output root is under `local_outputs_do_not_commit`;
- `trades.csv`, `equity_curve.csv`, ZIP, screenshots, root outputs, data-vault
  outputs, and uploaded outputs are forbidden;
- anti-lookahead and data-leakage controls are explicit;
- metrics policy forbids PF, winrate, drawdown, Sharpe, expectancy, PnL,
  equity curve, ranking, leaderboard, and optimization score;
- W-01/W-02 remain gates and were not touched;
- no edge/performance/profitability claim is made.

## 5. Expected Audit Decisions

Choose exactly one:

- `M1_TRAIN_ONLY_PROTOCOL_DESIGN_AUDIT_PASS_READY_FOR_OWNER_EXECUTION_PROMPT_DECISION`
- `M1_TRAIN_ONLY_PROTOCOL_DESIGN_AUDIT_PASS_WITH_WARNINGS`
- `AUDIT_BLOCKED_UNAUTHORIZED_FILE_SCOPE`
- `AUDIT_BLOCKED_EXECUTION_AUTHORIZATION_LEAK`
- `AUDIT_BLOCKED_FORBIDDEN_DATA_POLICY`
- `AUDIT_BLOCKED_MANIFEST_SCHEMA`
- `AUDIT_BLOCKED_W01_W02_GATES`
- `AUDIT_BLOCKED_OUTPUT_POLICY`
- `AUDIT_BLOCKED_LOOKAHEAD_OR_LEAKAGE_CONTROLS`
- `AUDIT_BLOCKED_REGISTRY_SCOPE`

Passing this future audit still must not authorize M1 execution. It may only
allow the owner to decide whether to draft a separate exact M1 execution prompt.

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
