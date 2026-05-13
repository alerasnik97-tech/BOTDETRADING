# MANIPULANTE 3.0 — RESEARCH CHARTER

## 1. Executive Mandate & Philosophy
- **Rejection of MANIPULANTE 2.0:** The core configuration of MANIPULANTE 2.0 is officially rejected. Blindly optimizing its original translation parameters is forbidden.
- **No Blind Continuity:** MANIPULANTE 3.0 represents a rigorous architectural separation of concerns rather than an unguided attempt to rescue previous backtests.
- **Core Hypothesis Re-Framing:** Separation into **HTF (Direction / Context / Bias Filter)** and **LTF (Entry / Confirmation / Execution)**.
- **Objective Evidence over Intuition:** While the discretionary premise ("liquidity sweep on key levels followed by lower-timeframe structure shift") holds conceptual promise, the quant system operates with zero trust. Edge must be programmatically isolated and statistically proven.

## 2. Institutional Execution Protocol
- **No Live Authorization:** A positive backtest or validation outcome does **not** authorize direct deployment to Demo, Funded, or Live accounts.
- **Mandatory Forward Demo Gate:** Any promising candidate discovered here must successfully pass an extensive real-time forward demo period prior to production consideration.
- **TEST Partition Governance:** The TEST partition is strictly out-of-sample (semi-OOS relative to overall design parameters but completely unoptimized). It evaluates each final candidate exactly once without retroactive selection loops.
- **Official Primary Metric:** Strategy performance optimization and selection are governed strictly by **`net_r`** (net profit/loss in risk units after deduplicating all institutional frictions).
- **Realistic Cost Model:** Inclusion of FTMO-compliant commissions ($5/lot round-turn) and multi-tier slippage stress testing is unconditionally enforced on every execution.
- **News Fail-Close Integrity:** Macroeconomic event handling enforces a rigid fail-close regime. Missing or unverified calendar data aborts the evaluation.
- **Data & Production Sovereignty:** Source parquet data is read-only. Production environments (`01_CORE_PRODUCTION`) remain isolated and untouched.
