# BO01 FIRST TRAIN-ONLY BACKTEST FRAMEWORK DESIGN REPORT V1

## 1. Status
**`BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_READY_FOR_EXTERNAL_AUDIT`**

The technical specification and design of the first controlled, train-only backtesting framework for candidate strategy `BO01` has been documented.

---

## 2. Scope
This is a strictly document-based design phase:
- **Markdown-Only**: Only structural specifications are written.
- **No Python Execution**: No code was executed.
- **No Scripts Run**: No automation pipelines were triggered.
- **No Data Loading**: No CSV datasets were loaded or read.
- **No Backtesting**: Zero trade simulations or PnL calculations were performed.
- **No Training**: No parameter optimization or sweeps were executed.
- **No Partition Intrusion**: Validation and holdout data are not authorized for access.
- **No Future Dates**: Years 2025 and 2026 are not authorized for access.

---

## 3. Evidence Used
The design incorporates the physical evidence produced in the prior M2 evaluation phase:
- **Previous Phase Status**: `M2_TRAIN_ONLY_STRUCTURAL_RETRY_EXECUTION_AUDIT_PASS_WITH_WARNINGS`
- **Candidate Signal Densities (Jan 1 to Mar 31, 2015)**:
  - **Strategy `BO01`**: `638` contract-valid signals generated across 41 trading days. This operational density is sufficient for design rationale and testing simulation mechanics.
  - **Strategy `MR02`**: `5` signals across 3 trading days. This extremely low density makes it unsuitable for a primary backtesting framework at this stage. It remains in strict observation status.
- **No Prior Performance**: No profitability metrics or equity curves have been computed to date.

---

## 4. Design Summary
The technical framework design defines bounded operational rules to ensure backtest realism and prevent overfitting:
- **Execution Model**:
  - Candle-by-candle row-stepping in UTC with strict chronological order.
  - Max 1 trade active at any time.
  - Max 1 trade per day (strictly executing on the first valid signal of the session).
  - Conservative same-bar resolution (`STOP-FIRST` policy).
- **Cost Model**:
  - Evaluation under three predefined cost profiles: Base (1.2 pips spread / 0.2 pips slippage), Conservative (1.62 pips spread / 0.5 pips slippage), and Stress (3.0 pips spread / 1.0 pip slippage).
  - Round-turn commission of $7.00 per Standard Lot applied across all profiles.
- **Risk Model**:
  - Strictly R-multiple based metrics (constant trade sizing). No account compounding or dynamic recovery/Martingale logic allowed.
- **Metrics Policy**:
  - Authorized strictly objective net R-multiple statistics, winrate, drawdown, and cost impact. All optimization rankings, curve-fitting parameters, or FTMO/prop readiness claims are banned.
- **Abort Conditions**:
  - Immediate fail-closed execution termination if data leakage occurs, M15 prepared train dataset is missing, strategy logic is altered, or any parameter sweep is attempted.

---

## 5. Decision
The framework design complies with the quant laboratory governance rules within the reviewed scope. It is **ready for an external read-only audit** of the BO01 first train-only backtesting framework design.

---

## 6. Allowed Next Step
- External read-only audit of the BO01 first train-only backtesting framework design documents.

---

## 7. Forbidden Next Steps
- NO immediate backtest execution.
- NO loading of market data.
- NO validation or holdout partition access.
- NO 2025 or 2026 data loading.
- NO parameter optimization or sweeps.
- NO live, demo, or FTMO environment deployment.
