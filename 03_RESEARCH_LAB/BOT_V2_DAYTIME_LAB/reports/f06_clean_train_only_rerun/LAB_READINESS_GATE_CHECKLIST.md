# LAB READINESS GATE CHECKLIST

## 1. Current Status

`NOT_READY_FOR_LAB_GREEN_LIGHT`

D1 and D2 are validated. D3, D4, and D5 now have institutional proposals, but they still require explicit owner approval. No adapter has been implemented. No F06 real run, backtest, validation, holdout, 2025, or 2026 data has been touched.

## 2. Current Score

CURRENT_SCORE_0_100: `60`

Score basis:

| Area | Weight | Current |
| :--- | ---: | ---: |
| Governance D1-D5 | 35 | 24 |
| Adapter design and tests | 25 | 0 |
| Train-only micro-run evidence | 25 | 0 |
| Safety and repository hygiene | 15 | 11 |
| Total | 100 | 60 |

The governance score is partial because D3-D5 are proposed, not owner-approved. Safety is not full because local ignored/untracked historical root clutter existed before this package and should be cleaned under a separate manifest-backed hygiene task before final lab entry.

## 3. Required Score For Lab

REQUIRED_SCORE_FOR_LAB: `90`

A score below 90 blocks lab green light.

## 4. Open Blockers

CURRENT_BLOCKERS:

| Blocker | Severity | Resolution required |
| :--- | :--- | :--- |
| D3 not owner-approved | HIGH | Owner approves the exact canonical params and hash preimage. |
| D4 not owner-approved | HIGH | Owner approves the explicit base/conservative/stress cost policy. |
| D5 not owner-approved | CRITICAL | Owner approves Option A future additive telemetry change request. |
| Adapter absent | CRITICAL | Future adapter/mocks/tests only after D3-D5 approval. |
| No train-only micro-run | CRITICAL | One future train-only run only after adapter audit passes. |
| No Claude/ChatGPT output audit | HIGH | Future output audit must pass after validator returns `READY_FOR_CLAUDE_AUDIT`. |
| Local ignored root clutter exists | MEDIUM | Separate hygiene task should move/archive or manifest forbidden root artifacts without contaminating this package. |

## 5. Gates Remaining

### A. Governance

| Item | Status |
| :--- | :--- |
| D1 pinned | DONE_WITH_WARNINGS |
| D2 pinned | DONE |
| D3 pinned | PROPOSED_PENDING_OWNER_APPROVAL |
| D4 pinned | PROPOSED_PENDING_OWNER_APPROVAL |
| D5 resolved | PROPOSED_PENDING_OWNER_APPROVAL |

### B. Adapter

| Item | Status |
| :--- | :--- |
| implemented with mocks | NOT_STARTED |
| tests pass | NOT_STARTED |
| no real run | TRUE |
| no validation/holdout/2025/2026 | TRUE |
| Claude audit pass | NOT_STARTED |

### C. Real Train-Only Micro-Run

| Item | Status |
| :--- | :--- |
| one run_id | NOT_STARTED |
| train-only exact months | NOT_STARTED |
| validator `READY_FOR_CLAUDE_AUDIT` | NOT_STARTED |
| Claude output audit | NOT_STARTED |
| no certification yet | TRUE |

### D. Lab Entry

| Item | Status |
| :--- | :--- |
| after adapter | NO |
| after micro-run | NO |
| after audit | NO |
| strategy testing still train-only | REQUIRED |
| holdout untouched | TRUE |

## 6. Definition of Green Light

Green light requires all of the following:

1. D3, D4, and D5 are owner-approved with no ambiguity.
2. D5 Option A has a separate approved change request if core telemetry is still required.
3. Future adapter work is mocked/tested before any real run.
4. Future train-only micro-run uses one run_id and exact approved train months only.
5. `validate_rebuild_outputs.py` returns `READY_FOR_CLAUDE_AUDIT` before any human interprets results.
6. External audit accepts the output contract, cost model, rowcounts, hashes, and no-leakage evidence.
7. Validation, holdout, 2025, and 2026 remain untouched.
8. No ZIP workflow or old V50B outputs are used as source of truth.

## 7. What Is Still Forbidden

- adapter implementation before owner approval
- F06 real execution now
- backtest now
- strategy execution now
- optimization or sweep
- validation
- holdout
- 2025
- 2026
- core modification now
- schema weakening
- `gross_r = net_r` proxy as official cost evidence
- using old V50B trades/ranking outputs
- ZIP source-of-truth workflow
- main branch push
- force push
- PR ready conversion
- certification

## 8. Final Decision

FINAL_DECISION: `OWNER_APPROVAL_REQUIRED_BEFORE_NEXT_GATE`

NEXT_GATE: `OWNER_APPROVAL_D3_D4_D5_THEN_DESIGN_ONLY_ADAPTER_AND_TELEMETRY_CHANGE_REQUEST`

GREEN_LIGHT_NOW: `NO`
