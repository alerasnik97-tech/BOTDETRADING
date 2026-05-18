# NEXT PROMPT — EXTERNAL READ-ONLY AUDIT OF POST-EXTREME GOVERNANCE HYGIENE PATCH V1

## 0. Nature Of This Document

This is a future prompt for an external, read-only institutional audit of the
post-extreme-audit governance/hygiene patch. It authorizes auditing only. It
does not authorize modifying code, tests, or data, and it does not authorize
any execution.

## 1. Activation

Use this prompt only after the owner explicitly requests an external
read-only audit of the governance/hygiene patch.

## 2. Scope To Audit

- branch: `governance/post-extreme-audit-hygiene-patch-v1-20260517`
- patch report: `POST_EXTREME_AUDIT_GOVERNANCE_HYGIENE_PATCH_REPORT_V1.md`
- the real diff of the patch commit (governance markdown only).
- BO01/MR02 registry rows and subsection 3.1 (W-03): correct state, no
  invented results, explicit no-edge/no-run statements, required governance
  fields present.
- TP-01 lineage note, subsection 3.2 (W-05): rejection unchanged, canonical
  classification intact, traceability note present, no invented commit.
- Registry Maintenance Protocol §2 correction (W-04): no owner-less path to
  `TRAIN_RUN_PENDING` or execution.
- Taxonomy (W-04): `IMPLEMENTED_TESTS_PENDING` owner-gated; new
  owner-gated states present; `MICRO_RUN_EXECUTION_PENDING` nine
  preconditions present; `MICRO_RUN_PENDING` owner-gated; no owner-less
  route to a micro-run preflight remains.
- Execution plan (W-04): Section 1A present; green tests / external audit do
  not authorize execution; owner + audit gates between Phase 2 and Phase 3;
  W-01/W-02 reconciliation required before execution.
- W-01/W-02 remediation plan: documented only, nothing touched.
- no code/test/data/strategy/engine/runner change.
- no execution authorization anywhere.
- static safety scan classification.
- git scope: only the authorized governance markdown files changed; no
  secrets; no new output debt; pre-existing W-01 dirty tree untouched.

## 3. Prohibited

- do not modify code;
- do not modify tests;
- do not modify data;
- do not modify BO01/MR02 code or tests;
- do not modify the governance files (audit only);
- no micro-run;
- no dry-run;
- no backtest;
- no formal train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no Sub-Batch 1B;
- no parallel writers;
- no `git add .`, no force push, no merge/rebase/reset --hard/clean/stash.

## 4. Permitted

- read files;
- `git status`, `git log`, `git show`, `git diff`, `git ls-files`;
- create exactly one markdown audit report under
  `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`;
- commit and push only that single audit report.

## 5. Required Checks

1. The diff contains only the three authorized governance markdown files
   plus the patch report and this prompt — nothing else.
2. No code, test, or data file is in the diff.
3. BO01/MR02 rows assert no edge/performance and carry no invented metrics.
4. TP-01 rejection is unchanged and the lineage note invents no commit.
5. The taxonomy and execution plan no longer permit an owner-less path to a
   micro-run preflight; design ≠ execution is explicit; the nine
   `MICRO_RUN_EXECUTION_PENDING` preconditions are present.
6. The patch authorizes no execution and makes no edge/performance/
   profitability claim.
7. Static safety scan: classify every hit as NEGATIVE_DECLARATION_OK /
   GOVERNANCE_TERM_OK / HISTORICAL_REFERENCE_OK or BLOCKER.
8. Git/output/security: no secrets, no new output debt, W-01 untouched.

## 6. Decision Options

- `POST_EXTREME_GOVERNANCE_PATCH_AUDIT_PASS_READY_FOR_OWNER_DECISION`
- `POST_EXTREME_GOVERNANCE_PATCH_AUDIT_PASS_WITH_WARNINGS`
- `AUDIT_BLOCKED_UNAUTHORIZED_FILE_SCOPE`
- `AUDIT_BLOCKED_STATIC_SAFETY_SCAN`
- `AUDIT_BLOCKED_PATCH_AUTHORIZES_EXECUTION`
- `AUDIT_BLOCKED_INVENTED_RESULTS_OR_LINEAGE`

## 7. Output Requirement

Create only one audit markdown report under
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`. It must explicitly state that
no micro-run, dry-run, backtest, formal train, validation, holdout,
2025/2026 access, optimization, sweep, or Sub-Batch 1B is authorized, and
that the only possible next step is an owner decision on whether to
commission a design-only micro-run protocol.
