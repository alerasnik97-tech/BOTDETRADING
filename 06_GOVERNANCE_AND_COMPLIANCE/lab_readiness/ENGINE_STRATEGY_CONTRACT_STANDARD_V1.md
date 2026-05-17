# ENGINE-STRATEGY CONTRACT STANDARD V1 (ESC-STD-V1)

**Document Reference:** GOV-ESC-V1-20260517  
**Status:** APPROVED & MANDATORY  
**Subject:** Formal Execution, Timezone, and Anti-Lookahead Specifications for Strategy Skeletons  

---

## 1. OBJECTIVE & MANDATORY APPLICATION

This standard defines the official contract of execution, timezone handling, and causal integrity that every strategy model must strictly adhere to within the Institutional Quant Research Lab. No strategy skeleton can be evaluated in formal backtesting, validation, or holdout environments without passing the automated contract guardrails enforcing this standard.

---

## 2. SIGNAL & EXECUTION CONTRACTS

Every strategy module must implement the required interface precisely to ensure high-fidelity execution:

### 2.1 Interface & Signature
The entry point for generating strategy decisions is the `signal` function, which must be declared exactly as:
```python
def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
```
*   `frame`: The complete pricing DataFrame passed by the engine.
*   `i`: The current evaluation index (0-indexed integer corresponding to the active bar).
*   `params`: A dictionary containing concrete strategy hyperparameters.

### 2.2 Decision Contract
The `signal` function must return either:
*   `None`: Indicating no entry signal or action for the active bar.
*   `dict[str, Any]`: Indicating an active signal, containing exactly:
    *   `"direction"`: `str` (`"long"` or `"short"`)
    *   `"entry_price"`: `float` (the target execution price, which must be equal to `frame["close"].iat[i]`, i.e., executing on the bar close)

### 2.3 Causal Bounds
The strategy must never inspect or mutate any index rows `j > i`. It must treat the pricing series beyond index `i` as completely non-existent to guarantee strict forward causality.

---

## 3. TIMEZONE & TEMPORAL CONTRACTS

To eliminate execution mismatches, alignment errors, and timestamp slippage across DST boundaries and weekend gaps, the platform enforces a strict localization contract.

### 3.1 Datetime Index Localization
*   All pricing series (M1, M5, or higher) must have a timezone-aware `pd.DatetimeIndex` localized exactly to the New York timezone (`"America/New_York"`).
*   Index transitions must handle standard EST/EDT transitions (Standard Time vs. Daylight Saving Time) causal-transparently, preserving 24-hour day boundaries without offset drift or overlapping bars.

### 3.2 Intraday Cadence Integrity
*   Intraday cadences (e.g., M1, M5, M15) must maintain high regularity. Median time diffs are computed using robust intraday statistics (filtering out large gaps > 1 hour, such as overnights and weekends).
*   Inferred cadences must match the declared target timeframe. Any discrepancy (such as loading M5 data while running under M1 declaration) triggers the official telemetry warning `WARN_DECLARED_TIMEFRAME_DIFFERS_FROM_EFFECTIVE_CADENCE` to prevent reporting distortions.

---

## 4. UNIVERSAL CAUSALITY & PROHIBITED PRACTICES

Any strategy that reads future information will be permanently disqualified. The platform employs an automated Future-Row Poisoning Harness to guarantee no-lookahead compliance.

### 4.1 Anti-Lookahead Poisoning Rule
During the contract pre-flight, the verification harness replaces future close prices with `np.nan` (NaN poisoning) for all rows `j > i`:
*   `frame.loc[frame.index[j], "close"] = np.nan` for all `j > i`.
*   If the strategy's returned `signal(frame, i, params)` changes in any way under poisoning compared to clean execution, a severe lookahead violation is raised, and the strategy is rejected with `ReconciliationGateError`.

### 4.2 Forbidden Lookahead Patterns
The following coding practices are strictly prohibited:
1.  **Direct Future Indexing:** Referencing `frame.iloc[i + 1]` or similar.
2.  **Unbounded Rolling Calculations:** Operating rolling calculations (such as `frame["close"].rolling(30).mean()`) directly within the `signal` loop if the calculation is performed on the entire multi-year DataFrame. *Self-Correction: All rolling precomputations must be vectorized outside the loop, or sliced securely to `:i` inside the loop.*
3.  **Global Series Metrics:** Referencing global metrics (like overall mean, max, standard deviations, or percentiles) computed over the entire duration of the backtest frame. All statistical thresholds must be rolling or cumulative up to bar `i`.

---

## 5. RECONCILIATION & GOVERNANCE COMPLIANCE

*   **Audit Frequency:** Automatically enforced on every targeted run and pull request.
*   **Gate Status:** Fail-closed. Any contract or lookahead violation halts runner execution and blocks report serialization.
*   **Waivers:** No strategy may bypass lookahead poisoning checks.

---
*End of Standard (ESC-STD-V1)*
