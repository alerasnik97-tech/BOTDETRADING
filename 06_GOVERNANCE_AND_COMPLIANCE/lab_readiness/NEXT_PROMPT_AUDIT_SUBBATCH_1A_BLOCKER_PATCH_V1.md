# NEXT PROMPT AUDIT SUBBATCH 1A BLOCKER PATCH V1

Act as an external read-only institutional audit committee for the Sub-Batch 1A BO01/MR02 blocker patch.

## Activation

Use this prompt only after the owner explicitly requests an external read-only audit of the Sub-Batch 1A blocker patch.

## Objective

Audit the actual diff and the committed patch for BO01/MR02 Asian range completeness and targeted tests. This audit must decide only whether the blocker patch is acceptable for owner review. It must not authorize execution.

## Scope To Audit

- BO01 range completeness patch.
- MR02 range completeness patch.
- missing `06:30` endpoint tests.
- duplicate timestamp tests.
- wrong cadence tests.
- BO01 short-side test.
- MR02 long-side test.
- MR02 third prior bar breach test.
- no-lookahead behavior.
- GMT/DST behavior.
- file scope.
- safety scan.
- staging and committed file list.
- no unauthorized files.

## Prohibited

- do not modify code;
- do not modify tests;
- do not modify data;
- do not execute micro-run;
- do not execute dry-run;
- do not execute backtest;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no registry, factory, engine, runner, or `strategies/__init__.py` changes;
- no Sub-Batch 1B.

## Permitted

- read files;
- use `git status`, `git show`, and `git diff`;
- run the lightweight BO01/MR02 unit/contract tests if the environment is safe;
- run the same lightweight related contract tests if still synthetic and bounded;
- create one markdown audit report;
- commit and push only the audit report.

## Required Checks

1. Verify that the diff contains only the approved strategy, test, and governance markdown files.
2. Verify that BO01 rejects missing `00:00`, missing `06:30`, duplicate timestamps, off-grid timestamps, wrong cadence, and missing expected M5 timestamps in the Asian range.
3. Verify that MR02 enforces the same Asian range completeness contract.
4. Verify that both strategies use only rows before `i` for Asian range construction.
5. Verify that no entry logic, fakeout logic, ATR/EMA logic, stop logic, target logic, params, registry, factory, engine, or runner behavior was changed outside the blocker.
6. Verify that the new tests are specific enough to fail on the prior count-only implementation and pass only after the completeness patch.
7. Verify GMT/DST tests still use synthetic data and do not depend on current date, internet, external files, or market data.
8. Run the static safety scan over the patch files and classify every hit as allowed governance wording or blocker.

## Decision Options

- `AUDIT_PASS_SUBBATCH_1A_BLOCKER_PATCH_READY_FOR_OWNER_REVIEW`
- `AUDIT_PASS_WITH_MINOR_WARNINGS_OWNER_REVIEW_REQUIRED`
- `AUDIT_BLOCKED_BO01_RANGE_COMPLETENESS`
- `AUDIT_BLOCKED_MR02_RANGE_COMPLETENESS`
- `AUDIT_BLOCKED_TEST_QUALITY_RISK`
- `AUDIT_BLOCKED_STATIC_SAFETY_SCAN`
- `AUDIT_BLOCKED_UNAUTHORIZED_FILE_SCOPE`
- `AUDIT_BLOCKED_OUTPUT_POLICY`

## Output Requirement

Create only one audit markdown report under `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`. The report must state that no micro-run, dry-run, backtest, train, validation, holdout, 2025/2026 access, optimization, sweep, or Sub-Batch 1B is authorized.
