# SUBBATCH 1A BO01/MR02 M1 TRAIN-ONLY CONTROLLED MICRO-RUN PROTOCOL DESIGN V1

## 1. Status

`M1_TRAIN_ONLY_PROTOCOL_DESIGN_READY_FOR_EXTERNAL_AUDIT`

This status means design-only. It does not authorize any execution.

## 2. Nature Of This Document

This is a markdown-only protocol design. It performs no execution and loads no
data.

Explicit prohibitions for this phase:

- no M1 execution;
- no M0 execution;
- no data loading;
- no backtest;
- no formal train;
- no dry-run;
- no validation;
- no holdout;
- no 2025/2026 data;
- no optimization/sweep;
- no grid search;
- no walk-forward;
- no Sub-Batch 1B;
- no parallel writers;
- no code changes;
- no test changes;
- no data changes;
- no edge/performance/profitability claims.

## 3. Purpose

M1, if later separately approved, would be a minimal train-only controlled
micro-run whose only purpose is to verify real-data plumbing for BO01/MR02.

Future M1 may verify only:

- BO01/MR02 can be called over the smallest practical permitted train-only
  EURUSD sample;
- the data adapter path is controlled and bounded;
- timestamp and timezone handling is sane;
- the required columns contract is satisfied or fails closed;
- no obvious lookahead plumbing exists;
- outputs stay inside the approved ignored output root;
- the manifest schema is complete enough for external audit;
- no forbidden data is accessed;
- no source files are modified.

M1 cannot prove edge, profitability, robustness, production readiness,
validation readiness, demo readiness, real-account readiness, or FTMO readiness.

## 4. Candidates

| Strategy | Family | Timeframe | Contract Notes | Current State |
| :-- | :-- | :-- | :-- | :-- |
| BO01 | London Breakout (`LBC`) | M5 | Requires `open`, `high`, `low`, `close`, `volume`, `ema_m15_200`; Asian range `00:00-06:30` UTC/GMT; entry `07:00-10:00`; fixed `target_rr=2.0`; `daily_trade_count` and `has_active_position` gates. | Skeleton + tests + M0 synthetic plumbing passed with warnings; no edge; no performance; no real-data run. |
| MR02 | London Fakeout Reversion (`LBF`) | M5 | Requires `open`, `high`, `low`, `close`; Asian range `00:00-06:30` UTC/GMT; entry `07:00-11:00`; fakeout breach and engulfing confirmation; fixed `target_rr=1.5`; `daily_trade_count` and `has_active_position` gates. | Skeleton + tests + M0 synthetic plumbing passed with warnings; no edge; no performance; no real-data run. |

Both strategies require exact GMT M5 Asian-range timestamp coverage from `00:00`
through `06:30` inclusive, one unique row per expected timestamp, and rejection
of missing endpoints, duplicates, wrong cadence, or non-finite inputs.

## 5. Data Policy M1

Allowed future data, only if separately approved:

- EURUSD only;
- train-only data only;
- date range maximum: `2015-01-01` through `2024-12-31`;
- smallest practical sample sufficient to test plumbing;
- no validation partition;
- no holdout partition;
- no 2025;
- no 2026;
- no future leakage;
- no data selection based on result;
- no optimization/sweep.

The expected future source identity is `EURUSD_PREPARED_TRAIN_2015_2024_M5`.
The protocol does not open that source in this phase. A future execution prompt
must re-verify source availability, hashes, row counts, path, and observed
min/max timestamps before any strategy call.

### M1A - Metadata/Data Availability Preflight Only

Future M1A, if separately approved, may inspect metadata only:

- path existence;
- file identity;
- row count;
- date min/max;
- column names;
- data source hash/manifest;
- train-only partition label.

M1A must not execute BO01/MR02. M1A must not compute price statistics beyond
row count, min/max timestamps, and column availability.

### M1B - Tiny Train-Only Controlled Execution

Future M1B, if separately approved after M1A and audit, may execute a tiny
train-only controlled call path:

- maximum sample: one predeclared small contiguous train-only slice, preferably
  one to three trading days or the minimum bars required for the Asian window,
  warmup, and one entry-window probe;
- no performance metrics;
- no trades/equity files;
- only plumbing, timestamp, column, fail-closed, and data-contract outcomes.

M1B must predeclare the date range before execution and must not change the
range based on results.

### M1C - Slightly Broader Train-Only Sample

Future M1C is allowed only if M1B passes external audit and the owner separately
approves it. It may use a slightly broader train-only sample, still bounded to
2015-2024, still EURUSD only, still no validation/holdout/2025/2026, and still
no optimization/sweep.

This document designs M1A/M1B/M1C only. It executes none of them.

## 6. W-01 / W-02 Gates

Before any future M1 execution:

W-01 must be resolved or quarantined:

- exact 11-file dirty tree must be documented;
- allowed current confinement:
  `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/`;
- no unrelated dirty files allowed;
- no staged files allowed;
- if W-01 changed since this design phase, abort.

W-02 must be governed:

- preexisting output debt must not be touched;
- output policy must define allowed outputs before execution;
- no root outputs;
- no ZIP;
- no `trades.csv` or `equity_curve.csv` committed;
- no cleanup of historical debt during M1 execution.

## 7. Manifest Schema Hardening

M1 must improve the M0 manifest. Required manifest fields:

- `run_id`;
- `phase`;
- `strategy_ids`;
- `branch`;
- `commit_sha`;
- `parent_commit_sha`;
- `repo_status_before`;
- `repo_status_after`;
- `python_version`;
- `timestamp_start_utc`;
- `timestamp_end_utc`;
- `data_policy_version`;
- `data_source_id`;
- `data_range_declared`;
- `data_range_observed`;
- `validation_used: false`;
- `holdout_used: false`;
- `used_2025_2026: false`;
- `optimization_sweep: false`;
- `output_root`;
- `created_files`;
- `sha256` for every output file;
- `manifest_sha256_external`;
- `command_log_sha256`;
- `report_sha256`;
- `data_access_log_sha256`;
- `no_secrets_detected`;
- `no_forbidden_outputs_detected`.

If the future manifest lacks branch/commit/self-hash style provenance, the
execution must produce at least `WARN_MANIFEST_PROVENANCE_GAP`. If any created
file lacks a hash, or if branch/commit cannot be verified externally, abort with
`BLOCKED_MANIFEST_INCOMPLETE`.

## 8. Future Output Policy

Future output root:

`03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/m1_train_only_bo01_mr02/<RUN_ID>/`

The output root must be gitignored before execution.

Allowed future local outputs only:

- `M1_TRAIN_ONLY_MICRORUN_REPORT.md`;
- `output_manifest.json`;
- `command_log.txt`;
- `data_access_log.txt`;
- optional small diagnostic JSON.

Forbidden outputs:

- `trades.csv`;
- `equity_curve.csv`;
- ZIP;
- screenshots;
- large files;
- root outputs;
- data vault outputs;
- GitHub upload of outputs.

Outputs must not be committed. Only governance markdown reports and prompts may
be committed after audit.

## 9. Future Command Policy

All command examples in this protocol are marked:

`DRAFT_DO_NOT_RUN - NON-EXECUTABLE TEMPLATE ONLY`

No command may be executed in this phase. A future M1 execution prompt must
separately approve the exact command set.

Non-executable template examples:

- `DRAFT_DO_NOT_RUN - NON-EXECUTABLE TEMPLATE ONLY`: verify Git state with
  `git status --short`.
- `DRAFT_DO_NOT_RUN - NON-EXECUTABLE TEMPLATE ONLY`: verify active Python
  processes with `Get-Process python -ErrorAction SilentlyContinue`.
- `DRAFT_DO_NOT_RUN - NON-EXECUTABLE TEMPLATE ONLY`: verify data source metadata
  only through a future audited metadata preflight.
- `DRAFT_DO_NOT_RUN - NON-EXECUTABLE TEMPLATE ONLY`: call BO01/MR02 only inside
  a future owner-approved M1B controlled runner.

Forbidden future commands:

- `formal_train_runner --execute`;
- validation runner;
- holdout runner;
- optimization runner;
- sweep/grid/walk-forward runner;
- commands touching 2025/2026;
- destructive Git commands.

## 10. Future M1 Success Criteria

M1 success, if later approved, may only mean:

- allowed train-only data loaded;
- BO01/MR02 call path works or fails closed;
- expected columns exist or missing columns are reported;
- timestamps are timezone-aware, monotonic, and inside the declared range;
- no forbidden data accessed;
- outputs contained;
- manifest complete;
- no source modifications.

M1 success must not mean:

- edge exists;
- strategy works economically;
- profitability;
- robustness;
- ready for backtest;
- ready for validation;
- ready for FTMO.

## 11. Future M1 Abort Conditions

Abort if:

- branch is `main`;
- dirty tree is not resolved or quarantined;
- staged files exist before execution;
- active Python research process exists;
- output root is not gitignored;
- data range includes 2025/2026;
- validation or holdout is accessed;
- optimization/sweep is requested;
- Sub-Batch 1B is requested;
- parallel writer is active;
- code/test/data modification is required;
- engine/runner/data_loader modification is required;
- data vault write is attempted;
- manifest cannot be completed.

## 12. Anti-Lookahead / Data Leakage Controls

Future M1 must verify:

- timezone-aware timestamps;
- UTC/GMT handling for the Asian range and entry windows;
- monotonic index;
- duplicate timestamp policy;
- exact M5 cadence for the required Asian range;
- no future bars accessed by the signal path;
- BO01 `ema_m15_200` is causal and available before BO01 is called, otherwise
  BO01 fails closed;
- ATR inputs are current/past only;
- no validation/holdout columns or labels;
- no result-based filtering;
- no parameter search;
- no date selection after observing outputs.

The future M1 audit should include a future-row poisoning or equivalent
causality check only if separately approved as part of M1 verification. This
design does not run that check.

## 13. Metrics Policy

Forbidden in M1:

- PF;
- winrate;
- drawdown;
- Sharpe;
- expectancy;
- PnL;
- equity curve;
- ranking;
- leaderboard;
- optimization score.

Allowed in M1:

- row count;
- date min/max;
- missing columns;
- exception counts;
- signal call count;
- `None` count;
- contract-valid return count;
- forbidden access count;
- output file count;
- manifest hash status.

## 14. Future Report Requirements

Future M1 report must declare:

- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no backtest;
- no formal train;
- no edge/performance claims;
- W-01 status;
- W-02 status;
- data source;
- declared data range;
- observed data range;
- manifest hash;
- outputs not committed.

## 15. External Audit Requirement

Before M1 execution:

- this protocol design must pass external read-only audit;
- the owner must approve an exact execution prompt;
- the execution prompt must be externally audited if material.

After M1 execution:

- execution outputs must be externally audited before any next step.

## 16. Allowed Next Step

External read-only audit of M1 train-only protocol design.

## 17. Forbidden Next Steps

- no M1 execution;
- no backtest;
- no train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no Sub-Batch 1B;
- no parallel writers;
- no production/demo/real/FTMO.
