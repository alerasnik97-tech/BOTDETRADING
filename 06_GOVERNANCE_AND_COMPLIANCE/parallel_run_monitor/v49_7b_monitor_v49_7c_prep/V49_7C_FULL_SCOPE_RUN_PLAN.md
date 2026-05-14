# V49.7C — FULL TRAIN/VAL COVERAGE RUN PLAN

## Goal
Execute a comprehensive, institutional-grade rerun of the R1 strategy using the repaired parameter grid with 100% temporal coverage for TRAIN and VAL periods.

## Required Scope
- **TRAIN**: 2020-01 to 2022-12 (36 months)
- **VAL**: 2023-01 to 2024-12 (24 months)
- **TEST**: 2025-01 to 2026-12 (**LOCKED / NO TOCAR**)

## Configuration Target
- **Min Configs**: 600
- **Ideal Configs**: 800-1000
- **Max Configs**: 1200
- **Deduped Grid**: Re-verify grid before run.

## Operational Protocol
1. **Preflight**: Distributed check of 50 random configs.
2. **Checkpoints**: Every 50 configs.
3. **Resume Logic**: Mandatory.
4. **Stress Test**: 0.3/0.5 pips slippage.
5. **Audits**: Rowcount, Metric Recalc, Duplicate, Temporal Concentration, Independent Verify.

## Governance
- **No V50 Authorization**: This run only certifies V49.7 edge.
- **No ZIP Workflow**: All outputs synced to GitHub.
- **Zero Test Leakage**: Fail-close on any OOS date detection.
