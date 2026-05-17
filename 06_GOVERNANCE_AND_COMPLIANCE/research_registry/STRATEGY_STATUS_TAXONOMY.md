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
-   **Required Evidence:** Unit test suite passing with 100% green status.
-   **Owner Approval Required:** No.
-   **Next Possible States:** `MICRO_RUN_PENDING`, `RETIRED`.

### `MICRO_RUN_PENDING`
-   **Meaning:** Technical checks passed. A 10-day dry-run preflight is authorized to confirm order execution.
-   **Allowed Actions:** Dry-run preflight execution (no `--execute` or limited temporal range).
-   **Forbidden Actions:** Full range (2015-2024) backtests.
-   **Required Evidence:** Dynamic preflight run console log or lightweight reports folder.
-   **Owner Approval Required:** No.
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
