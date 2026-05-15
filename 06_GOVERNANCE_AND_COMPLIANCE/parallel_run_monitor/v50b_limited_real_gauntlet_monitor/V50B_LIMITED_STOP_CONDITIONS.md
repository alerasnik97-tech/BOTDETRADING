# V50B LIMITED REAL GAUNTLET — STOP CONDITIONS

Immediate termination of V50B if:

1. **TEST Leakage**: Detection of any trade date >= 2025-01-01.
2. **Synthetic Patterns**: Detection of `dummy`, `synthetic` or `np.random` used for trade generation (not including config sampling).
3. **Core Drift**: Unauthorized changes to `src/v7_engine` or `src/v6_utils`.
4. **Data Mutation**: Write access detected in `05_MARKET_DATA_VAULT`.
5. **Memory Breach**: Process RAM > 12GB or CPU > 95% sustained for > 1 hour.
6. **Throttler Contamination**: Patterns suggesting engine state leak between configs.
7. **Hygiene Failure**: Creation of ZIP files or temporary root files.
8. **GitHub Violation**: Unauthorized push to `main` or `force push`.
