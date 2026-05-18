# STRATEGY STATUS TAXONOMY

This document establishes the official lifecycle states, transition gates, and permissions for systematic trading strategy candidates inside the laboratory. It forms the core of our anti-overfitting and anti-leakage governance architecture.

---

## 1. State Definition Matrix

### `IDEA_ONLY`
-   **Meaning:** Concept or hypothesis under discussion. No code or pre-registration exists.
-   **Allowed Actions:** Conceptual planning, math/statistical sketches.
-   **Forbidden Actions:** Code implementation, dataset loading, backtesting.
-   **Required Evidence:** Concept description in backlog or brainstorming logs.
-   **Owner Approval Required:** No.
-   **Next Possible States:** `PRE_REGISTERED`, `RETIRED`.

### `PRE_REGISTERED`
-   **Meaning:** Strategy pre-registered in the registry. Hypotheses, variables, and parameters are frozen prior to backtesting.
-   **Allowed Actions:** Writing strategy code skeleton, setting up unit tests.
-   **Forbidden Actions:** Running any historical data backtests or dry-runs.
-   **Required Evidence:** Frozen entry in `STRATEGY_RESEARCH_REGISTRY.md` and pre-registration template file.
-   **Owner Approval Required:** Yes, to authorize code implementation.
-   **Next Possible States:** `IMPLEMENTATION_APPROVAL_REQUIRED`, `RETIRED`.

### `IMPLEMENTATION_APPROVAL_REQUIRED`
-   **Meaning:** Code implementation is complete, pending validation of technical contracts.
-   **Allowed Actions:** Static code analysis, running standard unit test suite.
-   **Forbidden Actions:** Loading historical market data, running backtests.
-   **Required Evidence:** Strategy code file in `03_RESEARCH_LAB`, targeted contract unit tests added.
-   **Owner Approval Required:** Yes.
-   **Next Possible States:** `IMPLEMENTED_TESTS_PENDING`, `RETIRED`.

### `IMPLEMENTED_TESTS_PENDING`
-   **Meaning:** Strategy code has been checked, waiting to pass the targeted preflight contract unit tests.
-   **Allowed Actions:** Running `test_engine_strategy_contract.py` on the candidate strategy.
-   **Forbidden Actions:** Loading full historical data, running actual backtests.
-   **Required Evidence:** Unit test suite fully passing (all targeted contract tests green).
-   **Owner Approval Required:** Yes. Passing the contract tests does NOT authorize a micro-run, dry-run, backtest, or any execution. An explicit owner decision is required before any micro-run protocol design or any micro-run preflight.
-   **Next Possible States:** `IMPLEMENTED_TESTS_AUDITED_OWNER_PROTOCOL_DECISION_PENDING`, `RETIRED`.

### `IMPLEMENTED_TESTS_AUDITED_OWNER_PROTOCOL_DECISION_PENDING`
-   **Meaning:** Strategy code skeleton plus unit/contract tests are written AND have passed an external read-only audit. No micro-run, dry-run, backtest, or formal train exists. Awaiting an explicit owner decision on whether to commission a design-only micro-run protocol.
-   **Allowed Actions:** Read-only review of the skeleton, tests, and audit reports.
-   **Forbidden Actions:** Micro-run, dry-run, backtest, formal train, validation, holdout, 2025/2026, optimization, sweep, Sub-Batch 1B, parallel writers, any code/test/data change.
-   **Required Evidence:** External read-only audit report(s) for the skeleton/tests and blocker patch.
-   **Owner Approval Required:** Yes (mandatory gate). Reaching this state authorizes nothing beyond an owner decision.
-   **Next Possible States:** `MICRO_RUN_PROTOCOL_DESIGN_PENDING`, `RETIRED`.

### `MICRO_RUN_PROTOCOL_DESIGN_PENDING`
-   **Meaning:** The owner has explicitly approved commissioning a design-only micro-run protocol document. Designing is not running.
-   **Allowed Actions:** Authoring the micro-run protocol DESIGN markdown only (single writer).
-   **Forbidden Actions:** Any execution of any kind; micro-run; dry-run; backtest; formal train; validation; holdout; 2025/2026; optimization; sweep; code/test/data changes.
-   **Required Evidence:** Owner authorization recorded; design document drafted.
-   **Owner Approval Required:** Yes (already granted to enter this state).
-   **Next Possible States:** `MICRO_RUN_PROTOCOL_DESIGN_READY`, `RETIRED`.

### `MICRO_RUN_PROTOCOL_DESIGN_READY`
-   **Meaning:** The micro-run protocol design document is written and has passed a separate external read-only audit. This state does NOT authorize execution.
-   **Allowed Actions:** Read-only review of the audited design.
-   **Forbidden Actions:** Any micro-run/dry-run/backtest execution; validation; holdout; 2025/2026; optimization; sweep.
-   **Required Evidence:** External read-only audit report of the design document.
-   **Owner Approval Required:** Yes (a separate, explicit owner approval is required to move toward execution).
-   **Next Possible States:** `MICRO_RUN_EXECUTION_PENDING`, `RETIRED`.

### `MICRO_RUN_EXECUTION_PENDING`
-   **Meaning:** All design and approval preconditions are satisfied; a micro-run preflight may be scheduled under a separate execution prompt. Entering this state still does not auto-run anything.
-   **Allowed Actions:** Preparation of a separate, owner-approved, externally-audited execution prompt only.
-   **Required Preconditions (ALL mandatory):**
    1. an externally-audited micro-run protocol design;
    2. a separate external audit of that design;
    3. explicit owner approval for execution;
    4. a clean worktree, or documented quarantine of unrelated pre-existing dirty files (W-01);
    5. a defined output-policy gate (W-02) specifying allowed outputs, storage location, what is committed, what is git-ignored, and how trades/equity/ZIP are prevented from being staged;
    6. no holdout;
    7. no 2025/2026;
    8. no validation;
    9. no optimization/sweep.
-   **Forbidden Actions:** Executing the micro-run before all nine preconditions are met and separately audited.
-   **Required Evidence:** Owner approval record; design audit; output-policy gate definition; worktree/quarantine confirmation.
-   **Owner Approval Required:** Yes (mandatory, separate from all prior approvals).
-   **Next Possible States:** `TRAIN_RUN_PENDING`, `RETIRED`.

### `MICRO_RUN_PENDING`
-   **Meaning:** A micro-run preflight has been separately authorized via the gated path above (`MICRO_RUN_EXECUTION_PENDING`). It confirms order/stop/limit/telemetry plumbing only and proves nothing about edge.
-   **Allowed Actions:** Dry-run preflight execution strictly within a separately owner-approved, externally-audited protocol (no `--execute`, limited temporal range, synthetic or small controlled owner-approved data only).
-   **Forbidden Actions:** Full range (2015-2024) backtests; holdout; 2025/2026; validation; optimization; sweep; any inference of edge/performance from the preflight.
-   **Required Evidence:** Dynamic preflight run console log or lightweight reports folder, plus the satisfied `MICRO_RUN_EXECUTION_PENDING` preconditions.
-   **Owner Approval Required:** Yes (mandatory; the preflight is never auto-authorized by passing tests or audits).
-   **Next Possible States:** `TRAIN_RUN_PENDING`, `RETIRED`.

### `TRAIN_RUN_PENDING`
-   **Meaning:** Dry-run successful. Fully authorized for a one-shot, sealed backtest execution strictly on train data (2015–2024).
-   **Allowed Actions:** Official runner execution with `--execute` on train data.
-   **Forbidden Actions:** Running validation (2025/2026), sweeping/optimization parameters.
-   **Required Evidence:** Manifest and summary reports sealed (`sealed: True`, exit code 0).
-   **Owner Approval Required:** Yes.
-   **Next Possible States:** `TRAIN_GATE_GREEN_NEEDS_AUDIT`, `TRAIN_GATE_FAILED`.

### `TRAIN_GATE_FAILED`
-   **Meaning:** Train backtest executed but failed the technical reconciliation or showed catastrophic parameters.
-   **Allowed Actions:** Code debugging only.
-   **Forbidden Actions:** Validation, holdout, or paper trading.
-   **Required Evidence:** Run logs showing reconciliation failure or massive parameter mismatch.
-   **Owner Approval Required:** No.
-   **Next Possible States:** `WATCHLIST_NEEDS_REDESIGN`, `RETIRED`.

### `TRAIN_GATE_GREEN_NEEDS_AUDIT`
-   **Meaning:** Train backtest successfully completed, reconciled, and sealed. Pending external quant audit.
-   **Allowed Actions:** Read-only analysis of output configs, summaries, and yearly tables.
-   **Forbidden Actions:** Backtest rerun, code modification, validation or holdout exposure.
-   **Required Evidence:** Sealed output directory and post-run reconciliation report.
-   **Owner Approval Required:** No.
-   **Next Possible States:** `REJECTED_*`, `VALIDATION_APPROVAL_REQUIRED`.

### `WATCHLIST_LOW_SAMPLE`
-   **Meaning:** Strategy passed train gate but has low trade density (between 15 and 30 trades over 10 years).
-   **Allowed Actions:** Conceptual analysis of filters, watchlist tracking.
-   **Forbidden Actions:** Active trading, validation or holdout.
-   **Required Evidence:** Registry update documenting low sample characteristics.
-   **Owner Approval Required:** Yes.
-   **Next Possible States:** `WATCHLIST_NEEDS_REDESIGN`, `REJECTED_LOW_EDGE`.

### `WATCHLIST_COST_FRAGILE`
-   **Meaning:** Strategy is positive in base costs but destroyed under conservative or stress costs.
-   **Allowed Actions:** Conceptual spread/slippage sensitivity studies.
-   **Forbidden Actions:** Production deployment, validation.
-   **Required Evidence:** Cost profile summaries documenting sensitivity metrics.
-   **Owner Approval Required:** Yes.
-   **Next Possible States:** `WATCHLIST_NEEDS_REDESIGN`, `REJECTED_COST_FRAGILE`.

### `WATCHLIST_NEEDS_REDESIGN`
-   **Meaning:** Technical or cost fragility requires deep mathematical redesign of strategy entry/exit rules.
-   **Allowed Actions:** Conceptual redesign, writing a new pre-registration skeleton under a NEW strategy ID.
-   **Forbidden Actions:** Overwriting/modifying the rejected code script directly.
-   **Required Evidence:** New pre-registration template frozen.
-   **Owner Approval Required:** Yes.
-   **Next Possible States:** `PRE_REGISTERED`, `RETIRED`.

### `REJECTED_LOW_EDGE`
-   **Meaning:** Strategy showed clear negative expectancy, PF < 1.0, or high drawdowns under all cost profiles.
-   **Allowed Actions:** Archiving in logs, using as a negative control.
-   **Forbidden Actions:** Backtesting, optimizing, unsealing validation/holdout.
-   **Required Evidence:** Formal external audit report with rejection verdict.
-   **Owner Approval Required:** No.
-   **Next Possible States:** `RETIRED`.

### `REJECTED_REGIME_OBSOLETE`
-   **Meaning:** Strategy shows severe regime obsolescence (logging exactly zero trades since 2018 or similar).
-   **Allowed Actions:** Archiving in logs.
-   **Forbidden Actions:** Validation, holdout, or optimization.
-   **Required Evidence:** Temporal trade distribution analysis showing zero density.
-   **Owner Approval Required:** No.
-   **Next Possible States:** `RETIRED`.

### `REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE`
-   **Meaning:** Strategy combines negative expectancy and severe temporal trade concentration (e.g., TP-01).
-   **Allowed Actions:** Archiving in research logs, negative control.
-   **Forbidden Actions:** All active research actions on this candidate.
-   **Required Evidence:** Sealed external audit report.
-   **Owner Approval Required:** No.
-   **Next Possible States:** `RETIRED`.

### `VALIDATION_APPROVAL_REQUIRED`
-   **Meaning:** Strategy passed the train-only phase and external audit with outstanding statistics. Seeking owner authorization to run on validation data.
-   **Allowed Actions:** None (Waiting for explicit approval).
-   **Forbidden Actions:** Exposing validation data or unsealing files.
-   **Required Evidence:** Formal audit report with recommendation to advance.
-   **Owner Approval Required:** Yes (Mandatory gate).
-   **Next Possible States:** `VALIDATION_FAILED`, `VALIDATION_PASSED_NEEDS_AUDIT`, `RETIRED`.

### `RETIRED`
-   **Meaning:** Strategy is permanently archived and deactivated.
-   **Allowed Actions:** Viewing historical reports.
-   **Forbidden Actions:** All execution actions.
-   **Required Evidence:** Registry update.
-   **Owner Approval Required:** No.
-   **Next Possible States:** None.
