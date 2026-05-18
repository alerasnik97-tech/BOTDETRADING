# NEXT PROMPT SUBBATCH 1A TESTS AUDIT V1

Act as an external institutional read-only auditor for Sub-Batch 1A BO01/MR02 skeletons and tests.

## Activation

Audit only the implementation branch produced for Sub-Batch 1A skeletons and unit/contract tests. This prompt is read-only for code, tests, data, engine, runner, registry, and strategy implementation files.

## Required Audit Scope

Audit:

- actual diff;
- BO01 skeleton;
- MR02 skeleton;
- BO01 tests;
- MR02 tests;
- test quality;
- no-lookahead behavior;
- timezone/DST behavior;
- no 2025/2026;
- no holdout;
- no backtest;
- no micro-run;
- file scope;
- safety scan;
- staging;
- no unauthorized files.

## Prohibited

- do not modify code;
- do not modify tests;
- do not modify data;
- do not execute backtest;
- do not execute micro-run;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no engine changes;
- no runner changes;
- no registry changes;
- no strategy changes.

## Permitted

- read files;
- inspect `git status`, `git show`, and `git diff`;
- execute the same lightweight unit/contract tests if needed;
- create one markdown audit report;
- commit/push only that audit report if requested by the owner.

## Audit Checks

1. Confirm only whitelisted files were created or modified.
2. Confirm BO01 and MR02 use the repo module contract: `signal(frame, i, params)`, `default_params()`, `parameter_space()`, and `parameter_grid()`.
3. Confirm no registry or factory file was changed.
4. Confirm both skeletons use only rows up to index `i`.
5. Confirm future poisoning tests fail if rows after `i` influence the current signal.
6. Confirm GMT session windows are enforced independent of NY DST shifts.
7. Confirm fail-closed behavior for missing columns, tz-naive index, NaNs, insufficient history, daily trade count, and active position.
8. Confirm tests are synthetic, deterministic, and do not touch market data.
9. Confirm the static safety scan has no blocker hits.
10. Confirm the report does not declare performance, profitability, live readiness, or next-phase authorization.

## Output

Produce a markdown audit report with:

- status;
- files inspected;
- tests inspected or executed;
- safety scan classification;
- findings with severity;
- final decision limited to either read-only audit accepted or blocker identified;
- allowed next step limited to owner decision after audit.
