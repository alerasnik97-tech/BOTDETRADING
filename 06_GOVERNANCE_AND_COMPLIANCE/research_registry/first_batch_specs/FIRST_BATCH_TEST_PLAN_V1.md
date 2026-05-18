# FIRST BATCH TEST PLAN V1

This document establishes the official test designs and targeted unit/contract test requirements for the first batch strategies (`BO01`, `MR02`, `MR03`, `LS01`, `LS02`). **NO unit test files are currently written under `03_RESEARCH_LAB/research_lab/tests` and NO executions are authorized.**

---

## 1. Test Architecture Matrix

| Test Module | Purpose | Verification Focus | Target Strategy IDs |
| :--- | :--- | :--- | :--- |
| **`test_strategy_registration`** | Import and Registry Check | Verifies classes import cleanly and register in the central dictionary. | All |
| **`test_strategy_contract`** | Future Poisoning Scan | Validates that no logic accesses `i+k` closed bars or future rows. | All |
| **`test_strategy_tz`** | Timezone Window Guard | Checks session time boundary validations during daylight saving changes. | All |
| **`test_strategy_fills`** | Fills Invariance | Verifies entry spreads, stop limits, and commission calculations. | All |
| **`test_strategy_limits`** | Invariant Bounds | Assures daily maximum trade limits and concurrent limits. | All |

---

## 2. Test Specifications & Skeletons Design

### 1. Registry & Import Verification (`test_strategy_registration_[id].py`)
*   **Design Objective:** Confirm that the strategy class can be imported without compilation errors and matches the laboratory interface.
*   **Assertions Required:**
    ```python
    assert issubclass(StrategyClass, BaseStrategy)
    assert StrategyClass.ID == "[ID]"
    assert StrategyClass.FAMILY_ID == "[FAMILY_ID]"
    ```
*   **Execution Trigger:** Phase 2 (Specs approved by owner).

### 2. Timezone & DST Transition Verification (`test_strategy_tz_[id].py`)
*   **Design Objective:** Verify that daylight saving shifts (EST/EDT) do not shift London or NY market time windows.
*   **Mock Fixture:** 
    *   Synthesize H4/M15/M5 bars covering the US DST transition weekend in March (e.g., March 10-15) and October.
*   **Assertions Required:**
    ```python
    # Confirm that London Breakout is strictly triggered after 07:00 GMT regardless of local EST clock shifts
    assert signal_timestamp.hour == 7
    assert signal_timestamp.minute == 0
    ```

### 3. Future Poisoning / Lookahead Prevention Verification (`test_strategy_contract_[id].py`)
*   **Design Objective:** Guarantee that no indicator calculations or trade decisions access future indices.
*   **Mock Fixture:**
    *   Create a dataframe of 100 historical closed bars.
    *   Run signal calculations on index `i`.
    *   Mutate the values of rows `i+1` to `100` (future bars) with random noise.
*   **Assertions Required:**
    ```python
    # Confirm mutating future values does not change the signal output for the current row
    assert base_signal_output == poisoned_signal_output
    ```

### 4. Maximum Daily Limit & Concurrent Limits (`test_strategy_limits_[id].py`)
*   **Design Objective:** Confirm that the engine strictly blocks new trade triggers once the daily maximum limit is reached.
*   **Mock Fixture:**
    *   Synthesize a continuous sequence of breakout triggers on the same day.
*   **Assertions Required:**
    ```python
    # For BO01: Max daily trades is 1
    assert len(triggered_trades_today) <= 1
    # For MR03: Max daily trades is 2
    assert len(triggered_trades_today) <= 2
    ```

### 5. Cost Profile & Slippage Deductions (`test_strategy_fills_[id].py`)
*   **Design Objective:** Confirm that execution costs (spread, commissions) degrade equity curves exactly as modeled.
*   **Assertions Required:**
    ```python
    assert fill_price_long == ask_price
    assert fill_price_short == bid_price
    assert commission_deduction == commission_rate * trade_notional
    ```

---

## 3. Succinct Execution Guide
*   **Step 1:** Once the owner approves the technical specifications and decides the sub-batch, targeted test scripts will be created inside `03_RESEARCH_LAB/research_lab/tests/`.
*   **Step 2:** Test scripts must run synchronously and return green exit code 0 status before any micro-run preflight is authorized.
