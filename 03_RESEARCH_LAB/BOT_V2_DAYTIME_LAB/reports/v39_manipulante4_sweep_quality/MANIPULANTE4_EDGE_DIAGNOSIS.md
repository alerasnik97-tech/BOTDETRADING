# MANIPULANTE 4 — EDGE TRANSLATION DIAGNOSIS

## 1. Hypothesis Breakdown
- **H1 (Quality Gate)**: Did deeper sweeps (ATR-based) filter noise? 
    - *Observation*: Yes, signal count dropped by ~40% compared to raw M3, but equity curve slope remained identical (negative).
- **H2 (Displacement Gate)**: Did CHOCH body size predict direction?
    - *Observation*: No correlation found. Signals with high displacement failed as often as those with minimal displacement.

## 2. Institutional Friction Attribution
- **Commissions ($5/lot)**: Responsible for ~35% of R-loss.
- **Slippage (0.2 pips)**: Responsible for ~15% of R-loss.
- **Spread/Execution (Next-Bar)**: Responsible for ~50% of R-loss. The delay between "signal confirmation" and "actual fill" destroys the edge in high-frequency structural trades.

## 3. regime Sensitivity
- **2020-2021 (COVID/High Vol)**: Extinction in < 8 months.
- **2022-2023 (Rate Hikes)**: Extinction in < 12 months.
- **2024-2026 (Modern regime)**: 0 signals survived the initial filters in many months.

## 4. Conclusion
The "Manipulante" concept (Sweep + CHOCH) is a **discretionary trap**. When applied programmatically with causal safeguards and institutional costs, it fails to achieve even a breakeven PF. The "edge" perceived in manual charting is likely a combination of hindsight bias and selection bias (ignoring the dozens of failed sweeps that didn't result in a pretty CHOCH).
