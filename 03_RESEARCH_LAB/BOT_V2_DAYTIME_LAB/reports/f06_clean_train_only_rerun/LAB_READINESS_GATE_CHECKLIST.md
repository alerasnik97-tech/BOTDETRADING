# LAB READINESS GATE CHECKLIST

## 1. Current Status

`NOT_READY_FOR_LAB_GREEN_LIGHT`

D1 remains validated with warnings. D2 remains validated. D3, D4, and D5 are now owner-approved. This advances governance closure but does not authorize laboratory green light, adapter implementation, core modification, F06 real execution, backtest execution, validation, holdout, 2025, or 2026.

## 2. Current Score

PREVIOUS_SCORE_0_100: `68`

CURRENT_SCORE_0_100: `78`

This score is intentionally not inflated. Governance is closed and the scoped D5 telemetry implementation exists with synthetic behavior-neutral tests, but Claude telemetry audit, safe adapter implementation/audit, train-only micro-run, validator output, external output audit, and final hygiene are still missing.

Score basis:

| Area | Weight | Current | Status |
| :--- | ---: | ---: | :--- |
| Governance D1-D5 owner decisions | 55 | 55 | Complete, with D1 warning carried forward |
| Next-gate prompts and documentation | 8 | 8 | Complete |
| Scope safety preserved in this task | 5 | 5 | Complete |
| D5 telemetry implemented and locally tested | 10 | 10 | Complete, pending Claude audit |
| D5 telemetry Claude audit | 2 | 0 | Not started |
| Adapter mocks/tests/audit | 10 | 0 | Not started |
| Train-only micro-run and output audit | 7 | 0 | Not started |
| Local/repo hygiene final claim | 3 | 0 | Not complete |
| Total | 100 | 78 | No lab green light |

## 3. Required Score For Lab

REQUIRED_SCORE_FOR_LAB: `90`

A score below 90 blocks lab green light.

## 4. Open Blockers

CURRENT_BLOCKERS:

| Blocker | Severity | Resolution required |
| :--- | :--- | :--- |
| Telemetry audit not passed | CRITICAL | Claude audit must pass before adapter implementation. |
| Adapter absent | CRITICAL | Future adapter must be implemented with mocks/tests only after telemetry audit PASS. |
| Adapter audit not passed | CRITICAL | Claude audit must pass before any real run. |
| No train-only micro-run | CRITICAL | One future train-only run only after adapter audit PASS and explicit authorization. |
| Validator not run on real outputs | HIGH | Future outputs must return `READY_FOR_CLAUDE_AUDIT`. |
| No Claude/ChatGPT output audit | HIGH | Future output audit must pass before lab green light. |
| Local ignored root clutter exists | MEDIUM | Separate hygiene task should clean or manifest forbidden root artifacts before final lab readiness claim. |

## 5. Gates Remaining

### A. Governance

| Item | Status |
| :--- | :--- |
| D1 pinned | DONE_WITH_WARNINGS |
| D2 pinned | DONE |
| D3 pinned | DONE_OWNER_APPROVED_AND_HASHED |
| D4 pinned | DONE_OWNER_APPROVED |
| D5 resolved | DONE_OWNER_APPROVED_AS_FUTURE_CHANGE_REQUEST |

### B. D5 Telemetry

| Item | Status |
| :--- | :--- |
| separate change request opened | DONE_LOCAL_BRANCH |
| core telemetry implemented | DONE_LOCAL_SYNTHETIC_TESTED |
| behavior-neutral tests pass | DONE_LOCAL_SYNTHETIC_TESTED |
| ledger telemetry fields available | PARTIAL_GROSS_R_UNAVAILABLE_FLAGGED |
| Claude telemetry audit PASS | NOT_STARTED |

### C. Adapter

| Item | Status |
| :--- | :--- |
| use only after telemetry audit PASS | REQUIRED |
| implemented with mocks | NOT_STARTED |
| tests pass | NOT_STARTED |
| no real run | TRUE |
| no validation/holdout/2025/2026 | TRUE |
| Claude adapter audit PASS | NOT_STARTED |

### D. Real Train-Only Micro-Run

| Item | Status |
| :--- | :--- |
| one run_id | NOT_STARTED |
| train-only exact months | NOT_STARTED |
| validator `READY_FOR_CLAUDE_AUDIT` | NOT_STARTED |
| Claude output audit | NOT_STARTED |
| no certification yet | TRUE |

### E. Lab Entry

| Item | Status |
| :--- | :--- |
| after D5 telemetry audit | NO |
| after adapter audit | NO |
| after train-only micro-run | NO |
| after output audit | NO |
| strategy testing still train-only | REQUIRED |
| holdout untouched | TRUE |

## 6. Green Light Definition

ChatGPT may only give green light to the laboratory when all of the following are true:

1. D1-D5 approved.
2. D5 telemetry change request implemented and audited.
3. Adapter implemented with mocks/tests.
4. Adapter audit Claude PASS.
5. Train-only micro-run executed.
6. Validator `READY_FOR_CLAUDE_AUDIT`.
7. Claude output audit PASS.
8. No validation, holdout, 2025, or 2026 touched.
9. Local/repo hygiene clean.
10. PRs/docs updated.

## 7. What Is Still Forbidden

- adapter implementation before telemetry audit PASS
- core modification outside a scoped D5 telemetry change request
- F06 real execution now
- backtest now
- strategy execution now
- optimization or sweep
- validation
- holdout
- 2025
- 2026
- schema weakening
- `gross_r = net_r` proxy as official cost evidence
- using old V50B trades/ranking outputs
- ZIP source-of-truth workflow
- main branch push
- force push
- PR ready conversion
- merge
- certification

## 8. Final Decision

FINAL_DECISION: `D5_TELEMETRY_IMPLEMENTED_READY_FOR_CLAUDE_AUDIT_BUT_LAB_GREEN_LIGHT_BLOCKED`

NEXT_GATE: `CLAUDE_D5_TELEMETRY_AUDIT`

GREEN_LIGHT_NOW: `NO`
