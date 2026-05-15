# PHASE 3 ENGINE INTERFACE AUDIT

## 1. Context
- **Objective**: Record the real engine inventory before any Phase 3 safe adapter is designed.
- **Constraints**: No adapter implementation, no F06 real run, no backtest, no validation, no holdout, no 2025/2026, and no core modification.

## 2. Physical Engine Inventory
- `src/v7_engine/`: **NOT PRESENT** in this checkout at PR #6 head.
- `src/v6_utils/`: **NOT PRESENT** in this checkout at PR #6 head.
- Visible engine surface identified during Claude audit: `research_lab/engine.py`.
- Visible callable backtest surface: `research_lab.engine.run_backtest(...)`.
- Visible engine configuration surface: `research_lab/config.py`.

## 3. Decision
The prior report text that referenced `src/v7_engine` and `src/v6_utils` as present engine targets was inaccurate for this checkout. Adapter implementation remains blocked until a separate engine inventory / adapter design pass maps the real `research_lab/engine.py` API, expected dataframes, cost model behavior, and output contract generation path.

## 4. Next Step
After PR6 runner hardening passes re-audit, the next allowed step is **engine inventory / adapter design only**. A real F06 run remains forbidden until the adapter is implemented, independently audited, and explicitly approved.
