# RESEARCH REJECTION GATES

This document establishes the official quantitative thresholds and anti-overfitting rules for systematic trading strategy candidates inside the laboratory during the `TRAIN-ONLY` phase (2015–2024).

---

## 1. Quantitative Gating Thresholds

### Gate 1: Sample Size Gate
-   **Hard Reject Threshold:** $< 15$ total trades.
-   **Watchlist Threshold:** $\ge 15$ and $< 30$ total trades.
-   **Advance Threshold:** $\ge 30$ total trades.
-   **Justification:** Low trade density prevents robust statistical significance and increases vulnerability to luck/outliers.
-   **Action:** If $< 15$, hard reject as `REJECTED_LOW_SAMPLE`. If $15-30$, flag as `WATCHLIST_LOW_SAMPLE`.

### Gate 2: Active Years Gate
-   **Hard Reject Threshold:** $< 3$ distinct years with $\ge 3$ trades each.
-   **Watchlist Threshold:** $3-4$ distinct active years.
-   **Advance Threshold:** $\ge 5$ distinct active years.
-   **Justification:** Edge must demonstrate persistence across multiple market regimes and macro conditions.
-   **Action:** Reject if the strategy only trades in one or two isolated years.

### Gate 3: Temporal Concentration Gate
-   **Hard Reject Threshold:** $> 35\%$ of total trades occurring within a single calendar month.
-   **Watchlist Threshold:** $20\%-35\%$ peak monthly concentration.
-   **Advance Threshold:** $< 20\%$ peak monthly concentration.
-   **Justification:** High monthly trade concentration indicates that the strategy is highly dependent on a specific market anomaly or structural event, which is unlikely to recur stably.
-   **Action:** Reject if trade concentration violates thresholds.

### Gate 4: Long Zero-Trade Period Gate
-   **Hard Reject Threshold:** $> 18$ consecutive months of zero trades during the active training scope.
-   **Watchlist Threshold:** $12-18$ consecutive months of zero trades.
-   **Advance Threshold:** $< 12$ consecutive months of zero trades.
-   **Justification:** Long periods of complete silence are indicative of extreme regime obsolescence, rule tightness, or database alignment errors.
-   **Action:** Reject as `REJECTED_REGIME_OBSOLETE` if inactive for $> 18$ months.

### Gate 5: Profit Factor (PF) Base Gate
-   **Hard Reject Threshold:** $PF < 1.15$ under the Base cost profile.
-   **Watchlist Threshold:** $1.15 \le PF < 1.30$.
-   **Advance Threshold:** $PF \ge 1.30$.
-   **Justification:** A profit factor below 1.15 does not provide enough statistical padding to survive live execution friction, broker markup, and slippage.
-   **Action:** Reject as `REJECTED_LOW_EDGE` if base $PF < 1.15$.

### Gate 6: Profit Factor (PF) Stress Gate
-   **Hard Reject Threshold:** $PF < 1.00$ under the Stress cost profile.
-   **Watchlist Threshold:** $1.00 \le PF < 1.10$.
-   **Advance Threshold:** $PF \ge 1.10$.
-   **Justification:** The strategy must remain net-positive even under extreme latency slippage and spread markups.
-   **Action:** Reject as `REJECTED_COST_FRAGILE` if stress $PF < 1.00$.

### Gate 7: Expectancy Gate
-   **Hard Reject Threshold:** $< 0.15$ R net expectancy per trade under the Base cost profile.
-   **Watchlist Threshold:** $0.15 \le Expectancy < 0.25$ R.
-   **Advance Threshold:** $Expectancy \ge 0.25$ R.
-   **Justification:** Low expectancy per trade means the strategy is highly sensitive to execution speed and spreads. Survival requires a minimum net safety buffer.
-   **Action:** Reject if base expectancy is $< 0.15$ R.

### Gate 8: Drawdown Gate
-   **Hard Reject Threshold:** $> 15\%$ maximum drawdown under the Base cost profile (assuming $0.5\%$ risk per trade).
-   **Watchlist Threshold:** $10\%-15\%$ maximum drawdown.
-   **Advance Threshold:** $< 10\%$ maximum drawdown.
-   **Justification:** High drawdowns relative to conservative risk settings prove that the strategy has high volatility and poor downside control.
-   **Action:** Reject if drawdown exceeds 15%.

### Gate 9: Cost Degradation Gate
-   **Hard Reject Threshold:** Expectancy degradation from Base to Stress $> 40\%$.
-   **Watchlist Threshold:** $25\%-40\%$ degradation.
-   **Advance Threshold:** $< 25\%$ degradation.
-   **Justification:** If stress costs degrade expectancy by more than 40%, the edge is purely theoretical and cannot be safely traded on retail or prop firm accounts.
-   **Action:** Reject as `REJECTED_COST_FRAGILE` if highly sensitive to transaction costs.

---

## 2. Platform & Safety Gates

### Gate 10: Metric Reconciliation Gate
-   **Requirement:** 100% mathematical matching between ledger and report summaries. Sealed manifest file (`sealed: True`, exit code 0).
-   **Action:** If any discrepancy or violation occurs, the run is immediately aborted and blocked.

### Gate 11: Output Policy Gate
-   **Requirement:** Heavy output files (`trades.csv`, `equity_curve.csv`) must reside strictly inside `local_outputs_do_not_commit` directories and never be staged or versioned in Git.
-   **Action:** Block commit and flag a governance violation if heavy files are staged.

### Gate 12: Holdout Protection Gate
-   **Requirement:** Date boundaries strictly capped at train-only (`2015-01-01` to `2024-12-31`). No reading or unsealing of 2025/2026 data files.
-   **Action:** Block research stream immediately if holdout data is contaminated.

---

## 3. No Optimization Rescue Rule
If a strategy candidate fails to pass any of the quantitative gates listed above, it is permanently **REJECTED**.
1.  **NO Optimization:** Researchers are strictly forbidden from modifying parameters, indicators, or rules in the failed code script to "rescue" it on the same training dataset.
2.  **Hypothesis Separation:** To test a new filter or modified concept, a new entry must be added to the registry with a unique strategy ID. The code must be written as a separate file, ensuring a completely fresh, traceable pre-registration audit trail.
