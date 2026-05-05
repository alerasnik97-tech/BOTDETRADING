# Comparison to Prior Phases - Phase 50K reproduction

## Summary of Results

| Phase | PF Tick | Expectancy Tick | Total R Tick | Winrate Tick | Match Rate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 50H** | ~4.66 | ~0.52R | - | - | - | **INVALIDATED (Probable Mirage)** |
| **Phase 50I** | ~0.049 | -0.89R | - | - | - | **CONFIRMED** |
| **Phase 50J** | 0.036 | -0.91R | -140.22R | 4.54% | - | **CONFIRMED** |
| **Phase 50K (Indep)** | **0.047** | **-0.85R** | **-145.17R** | **5.32%** | **2.21%** | **REPRODUCED (Independent)** |

## Findings

1. **Reproduction Confirmed**: Phase 50K independently reproduces the negative results of Phase 50I and 50J. The strategy exhibits extreme negative expectancy on tick data.
2. **PF Tick Degradation**: The drop from PF 4.66 (50H) to 0.047 (50K) is cataclysmic. This indicates that the 50H results were likely generated using flawed logic (e.g., zero spread, mid-price execution, or incorrect TP/SL priority).
3. **Match Rate**: A match rate of 2.21% indicates that the bar-based simulation is almost entirely disconnected from tick-level reality for this specific strategy.
4. **Data Integrity**: Phase 50K used the official months and certified Parquets. 2025-08 was excluded and 2024-06 was included.
5. **Verdict**: The "Manipulante" edge is an artifact of low-resolution data and does not survive institutional-grade tick validation.

## Logical Interpretation

- **Phase 50H**: Likely used "too good to be true" assumptions.
- **Phase 50J**: Correctly identified the month contamination, but the core performance issue was already present.
- **Phase 50K**: Validates that even with correct months and strict rules, the strategy is not viable.
