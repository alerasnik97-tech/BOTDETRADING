# NEXT PROMPT — OWNER DECIDES: RETRY M2 CONSERVATIVE STRUCTURAL EXECUTION AFTER RUNNER AUDIT

## Context

The M2 Structural Runner has been created and externally audited:

- **Runner**: `research/m2-structural-runner-bo01-mr02-v2-20260518` @ `ac34c8a82bd44bc18bd17600385687efbe48d7b6`
- **Audit**: `audit/m2-structural-runner-bo01-mr02-v1-20260518`
- **Audit Result**: `M2_STRUCTURAL_RUNNER_AUDIT_PASS_WITH_WARNINGS`
- **Warnings**: W-01 (valid_signal_count semantics), W-02 (shallow test self-read)
- **Blockers**: 0

The audit does NOT prove edge, rentability, or strategy readiness.

---

## Owner Decision Required

The owner must choose one of the following options by issuing an explicit, autonomous authorization phrase:

---

### Option A — RETRY M2 CONSERVATIVE STRUCTURAL EXECUTION

If the owner wishes to proceed with M2 Conservative structural evaluation using the audited runner:

Issue the following phrase **exactly**, as an autonomous declaration (not in a quote, not as a log, not as an example):

```
APRUEBO REINTENTAR M2 CONSERVATIVE STRUCTURAL EXECUTION BO01/MR02 USANDO EL RUNNER AUDITADO, SIN CALCULAR PERFORMANCE, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.
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

### Option B — PATCH W-01 BEFORE RETRY

If the owner prefers to patch the minor `valid_signal_count` semantic issue before proceeding:

Issue the following phrase **exactly**:

```
APRUEBO MICRO-PATCH W-01 DEL RUNNER M2 STRUCTURAL ANTES DE REINTENTAR M2, SIN EJECUTAR M2, SIN CARGAR DATOS REALES, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.
```

What this authorizes:
- Moving `valid_signal_count +=1` to after successful `contract_valid_count` confirmation in the runner
- Re-running synthetic tests to confirm the fix
- Creating a new governance report and audit prompt

---

### Option C — PAUSE

If the owner wishes to pause and review before deciding:

No phrase needed. Simply do not issue Option A or B.

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
