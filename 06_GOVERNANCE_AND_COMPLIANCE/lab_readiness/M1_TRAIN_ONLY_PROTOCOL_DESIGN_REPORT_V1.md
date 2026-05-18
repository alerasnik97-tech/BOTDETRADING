# M1 TRAIN-ONLY PROTOCOL DESIGN REPORT V1

## 1. Status

`M1_TRAIN_ONLY_PROTOCOL_DESIGN_READY_FOR_EXTERNAL_AUDIT`

## 2. Scope

Markdown only.

- no code;
- no tests;
- no data;
- no execution;
- no backtest;
- no train;
- no dry-run;
- no validation;
- no holdout;
- no 2025/2026 data;
- no optimization/sweep;
- no Sub-Batch 1B;
- no parallel writers.

## 3. Files Created/Modified

Created:

- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/microrun_protocols/SUBBATCH_1A_BO01_MR02_M1_TRAIN_ONLY_PROTOCOL_DESIGN_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_TRAIN_ONLY_PROTOCOL_DESIGN_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M1_TRAIN_ONLY_PROTOCOL_DESIGN_V1.md`

Modified:

- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/STRATEGY_RESEARCH_REGISTRY.md`

Registry scope: BO01/MR02 only. The registry `Latest Commit` cell is recorded as
`BRANCH_HEAD` to avoid embedding a self-referential commit SHA inside the same
commit that creates this design state.

No code, tests, data, runner, engine, data_loader, registry/factory Python,
`strategies/__init__.py`, production, incubation, data vault, local outputs, or
ZIP files were modified.

## 4. M1 Protocol Summary

The M1 design is a future train-only controlled micro-run protocol for BO01/MR02.
It is not a run.

Sub-phases:

- M1A: metadata/data availability preflight only. No strategy execution. No
  price statistics beyond row count, date min/max, and column names.
- M1B: tiny train-only controlled execution, only if separately approved and
  audited. It may verify data contract, timestamp/timezone handling, call path,
  fail-closed behavior, output containment, and manifest completeness.
- M1C: slightly broader train-only sample only if M1B passes audit and the
  owner separately approves it. Still no validation, holdout, 2025/2026, or
  optimization/sweep.

BO01/MR02 remain skeleton-stage candidates with M0 synthetic plumbing passed
with warnings. No edge, performance, profitability, champion, demo, real, or
FTMO status is asserted.

## 5. Manifest Hardening

The protocol corrects `WARN_MANIFEST_MINOR_GAP` by requiring future M1 manifests
to include:

- branch;
- commit SHA;
- parent commit SHA;
- repo status before/after;
- data policy version;
- data source ID;
- declared and observed data ranges;
- explicit `validation_used: false`;
- explicit `holdout_used: false`;
- explicit `used_2025_2026: false`;
- explicit `optimization_sweep: false`;
- SHA256 for every output file;
- `manifest_sha256_external`;
- `command_log_sha256`;
- `report_sha256`;
- `data_access_log_sha256`;
- forbidden-output and secret checks.

If branch/commit/self-hash style provenance is absent in a future M1 manifest,
the future M1 result must warn or abort according to severity.

## 6. W-01/W-02 Handling

W-01 and W-02 were not touched.

W-01 remains a future execution gate:

- current known dirty tree is 11 files under
  `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/`;
- before any future execution it must be resolved or formally quarantined;
- any drift outside that confinement is an abort condition.

W-02 remains a future execution gate:

- preexisting output debt must not be touched by M1;
- output policy must define allowed outputs before execution;
- no root outputs;
- no ZIP;
- no `trades.csv` or `equity_curve.csv` committed.

## 7. Decision

Ready for external read-only audit.

This report does not authorize M1 execution, backtest, train, dry-run,
validation, holdout, 2025/2026, optimization/sweep, Sub-Batch 1B, parallel
writers, production, demo, real, or FTMO.

## 8. Allowed Next Step

External read-only audit of M1 train-only protocol design.

## 9. Forbidden Next Steps

- no M1 execution;
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
- no edge/performance/profitability claims.
