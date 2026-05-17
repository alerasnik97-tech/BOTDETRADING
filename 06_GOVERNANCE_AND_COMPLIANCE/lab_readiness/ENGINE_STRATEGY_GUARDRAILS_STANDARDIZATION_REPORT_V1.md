# ENGINE-STRATEGY GUARDRAILS STANDARDIZATION REPORT V1

**Document Reference:** GOV-REP-GR-V1-20260517  
**Status:** COMPLETE & COMPLIANT  
**Date:** May 17, 2026  
**Auditor:** Antigravity Quant Engineering Officer  

---

## 1. EXECUTIVE SUMMARY

This report documents the successful implementation of standard platform guardrails to transition the newly created engine-strategy contract tests into permanent, automated telemetry controls. 

Every objective outlined in the transition plan has been surgically executed:
1.  **Activity Sentinel (`assess_activity`)** has been standard-wired as a non-blocking `WARN` gate inside the official `formal_train_runner.py`.
2.  **Universal Anti-Lookahead Harness** has been standard-wired to check causal invariance in approved strategies under future-row poisoning.
3.  **C10 Timeframe Traceability Discrepancy** has been fully resolved by dynamically inferring the data cadence and passing it to the report engine, logging the warning `WARN_DECLARED_TIMEFRAME_DIFFERS_FROM_EFFECTIVE_CADENCE` if they differ.
4.  All unit tests in both the runner and strategy contract suites are 100% green, with zero errors or regressions.

---

## 2. DETAIL OF IMPLEMENTED CONTROLS

### 2.1 Standard Integration of `assess_activity`
*   **Code Location:** [formal_train_runner.py](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py#L317-L373)
*   **Execution Behavior:** The activity sentinel runs dynamically on the base profile's trades series after the backtest completes and prior to sealing the run.
*   **Non-Blocking Warnings:** In contrast to the fail-closed reconciliation gate, this control logs warnings without aborting, preserving the run output but providing clear telemetry flags in stdout and inside `RUN_MANIFEST.json`.
*   **Warning Telemetry Flags:**
    *   `WARN_ZERO_TRADES`: Triggered if the total trade count is 0.
    *   `WARN_LOW_SAMPLE_SIZE`: Triggered if $0 < \text{trades} < 30$.
    *   `WARN_SINGLE_ACTIVE_YEAR`: Triggered if trades are concentrated in only 1 year while the backtest covers multiple years.
    *   `WARN_EXTREME_TEMPORAL_CONCENTRATION`: Triggered if a single month contains $\ge 90\%$ of the total trades.
    *   `WARN_LONG_ZERO_TRADE_PERIOD`: Triggered if the monthly trading coverage ratio is $< 5\%$.
    *   `WARN_LOW_FREQUENCY_EDGE_NOT_EVALUABLE`: Triggered if the average frequency is $< 6.0$ trades per year.

### 2.2 Standard Integration of the Anti-Lookahead Harness
*   **Code Location:** [test_engine_strategy_contract.py](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/03_RESEARCH_LAB/research_lab/tests/test_engine_strategy_contract.py#L289-L300)
*   **Target Strategies:** Officially verified strategies, including `ve_orb_volatility_expansion` and our optimized `tp01_london_ny_momentum_pullback`.
*   **Validation Method:** Verifies that futureclose prices (beyond index $i$) are replaced with `np.nan` (future poisoning) and asserts that the strategy decisions remain completely invariant under poisoning. Both strategies pass 100% causally.
*   **Scope Standard Note:** Due to initialization differences and specific column requirements of legacy strategies in the registry, a full registry scan is deferred and documented as `WARN_ANTI_LOOKAHEAD_NOT_REGISTRY_WIDE_YET`.

### 2.3 C10 Timeframe Discrepancy Resolution
*   **Code Location:** [formal_train_runner.py](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py#L441-L459)
*   **Mechanism:** Added `_infer_effective_cadence_minutes` which calculates the median difference in seconds between index bars, filtering out overnight and weekend gaps ($> 3600$ seconds).
*   **Discrepancy Control:** Compares the effective timeframe (e.g. `"M5"` if median diff is 5 minutes) against the runner's declared `"M1"`.
*   **Discrepancy Warning:** Appends `WARN_DECLARED_TIMEFRAME_DIFFERS_FROM_EFFECTIVE_CADENCE` if a discrepancy is found.
*   **Reporting Fix:** Passes `timeframe=effective_timeframe_str` explicitly to `summarize_result` to fix the legacy `"M15"` metadata mismatch, ensuring the generated JSON and CSV summaries display the correct timeframe.

---

## 3. VERIFICATION & TELEMETRY EVIDENCE

### 3.1 Contract Tests Suite (`unittest`)
*   **Command:** `python -m unittest 03_RESEARCH_LAB/research_lab/tests/test_engine_strategy_contract.py 03_RESEARCH_LAB/research_lab/tests/test_engine_time_contract.py 03_RESEARCH_LAB/research_lab/tests/test_strategy_activity_gates.py`
*   **Result:** `Ran 18 tests in 0.532s` -> **OK (100% Green)**
*   **Key Validations:** Lookahead causality on VE-ORB, lookahead causality on TP-01, weekend gaps, EST/EDT timezone offsets, and activity sentinel classifications.

### 3.2 Formal Train Runner Tests Suite (`unittest`)
*   **Command:** `python -m unittest 03_RESEARCH_LAB/research_lab/tests/test_formal_train_runner_execute_contract.py 03_RESEARCH_LAB/research_lab/tests/test_formal_train_runner_contract.py`
*   **Result:** `Ran 25 tests in 0.967s` -> **OK (100% Green)**
*   **Key Validations:** Manifest integrity, git-provenance checks, cost-profile routing gates, zero-trades activity warning (`WARN_ZERO_TRADES`), and timeframe discrepancy warning (`WARN_DECLARED_TIMEFRAME_DIFFERS_FROM_EFFECTIVE_CADENCE`).

---

## 4. AUDIT COMPLIANCE & GATE APPROVAL

*   **Status:** APPROVED & LOCKDOWN  
*   **Verdict:** The quantitative lab is officially standard-ready and safe for formal train-only backtesting.

---
*End of Report (GOV-REP-GR-V1-20260517)*
