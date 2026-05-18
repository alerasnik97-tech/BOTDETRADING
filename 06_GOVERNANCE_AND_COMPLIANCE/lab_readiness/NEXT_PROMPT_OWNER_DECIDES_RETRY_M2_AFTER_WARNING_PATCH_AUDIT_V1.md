# NEXT PROMPT — OWNER DECIDES: RETRY M2 CONSERVATIVE STRUCTURAL EXECUTION AFTER WARNING PATCH AUDIT

## Context

The M2 Structural Runner Warning Patch has been applied and externally audited:

- **Runner**: `research/m2-structural-runner-warning-patch-v1-20260518` @ `62851977c223889d16107e74393076d41d88b315`
- **Audit**: `audit/m2-structural-runner-warning-patch-v1-20260518`
- **Audit Result**: `M2_STRUCTURAL_RUNNER_WARNING_PATCH_AUDIT_PASS_READY_FOR_M2_RETRY_OWNER_DECISION`
- **Warnings**: 0
- **Blockers**: 0

The audit does NOT prove edge, rentability, or strategy readiness.

---

## Owner Decision Required

The owner must choose one of the following options by issuing an explicit, autonomous authorization phrase:

---

### Option A — RETRY M2 CONSERVATIVE STRUCTURAL EXECUTION

If the owner wishes to proceed with M2 Conservative structural evaluation using the fully warning-patched and audited runner:

Issue the following phrase **exactly**, as an autonomous declaration (not in a quote, not as a log, not as an example):

```
APRUEBO REINTENTAR M2 CONSERVATIVE STRUCTURAL EXECUTION BO01/MR02 USANDO EL RUNNER PARCHEADO Y AUDITADO, SIN CALCULAR PERFORMANCE, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.
```

What this authorizes:
- Structural signal counting on train-only data (2015–2024)
- Use of `m2_structural_runner.run_m2_structural_evaluation` with BO01 and MR02
- Loading EURUSD M5 train-only parquet from `05_MARKET_DATA_VAULT` (read-only)
- Output: structural counts only (no PnL, no Profit Factor, no Sharpe, no winrate, no drawdown)

What this does NOT authorize:
- Performance metric calculation
- Validation or holdout data
- 2025 or 2026 data
- Optimization or parameter sweep
- Demo, real, or FTMO connection

---

### Option B — PAUSE

If the owner wishes to pause and review before deciding:

No phrase needed. Simply do not issue the Option A phrase.

---

## Strict Prohibitions for the Agent Executing This Prompt

- NO M2 execution without the exact Option A phrase
- NO loading real market data without Option A phrase
- NO backtest, train, validation, holdout
- NO 2025 or 2026 data
- NO optimization, sweep, grid search
- NO performance metrics (PnL, PF, Sharpe, winrate, drawdown, expectancy)
- NO modifying strategy code, engine, core
- NO declaration of edge or robustness
- NO push to main
- NO force push, reset --hard, git clean, git stash, git add .
