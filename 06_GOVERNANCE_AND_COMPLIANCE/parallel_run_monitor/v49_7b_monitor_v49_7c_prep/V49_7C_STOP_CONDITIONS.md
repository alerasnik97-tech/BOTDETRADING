# V49.7C — STOP CONDITIONS

Immediate termination of V49.7C if:

1. **TEST Leakage**: Any date >= 2025-01-01 detected in trade outputs.
2. **Core Drift**: Modification detected in `src/v7_engine` or `src/v6_utils`.
3. **Data Mutation**: Write access detected in `05_MARKET_DATA_VAULT`.
4. **Invalid Rowcount**: Mismatch between processed configs and output rows.
5. **Metric Failure**: Inability to recalculate PF from raw trades.
6. **Low Scope**: Config count falls below 600.
7. **Injection Regression**: Evidence of parameter injection (redundant trade sets).
8. **Resource Exhaustion**: Disk space < 5GB or RAM usage > 90% persistent.
9. **Hygiene Violation**: Creation of ZIP files or root-level temporary files.
10. **Git Violation**: Unauthorized push to `main` or `force push`.
