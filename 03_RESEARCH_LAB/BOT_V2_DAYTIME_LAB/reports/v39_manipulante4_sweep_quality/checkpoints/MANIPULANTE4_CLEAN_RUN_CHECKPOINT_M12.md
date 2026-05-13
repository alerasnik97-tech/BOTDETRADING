# MANIPULANTE 4 — CLEAN RUN CHECKPOINT [12/76]

## 1. Integrity Status
- **Status**: `CONTINUE_CLEAN`
- **Runner SHA256**: `60e78830f812183ecb12ce7b6a8743c73669f1e5d88176d0745d1ca7ca6aa92f`
- **Runner Frozen**: SÍ (No edits since restart).
- **Audit Date**: 2026-05-13 12:15 NY
- **Dataset Phase**: TRAIN (Month 12/76 completed, Month 17 in progress).

## 2. Constraint Verification
| Constraint | Status | Notes |
| :--- | :--- | :--- |
| EURUSD Only | PASS | |
| 07:00-17:00 NY | PASS | Exit at 16:55 NY verified. |
| Max 3 trades/day | PASS | 0 violations detected (divide by 2 slippages). |
| News Fail-Close | PASS | Handled by UnifiedV7Engine. |
| Rollover Blocked | PASS | 16:55-17:15 blocked. |
| Slippage 0.2 | PASS | |
| Comisión FTMO | PASS | |
| EOM Excluded | PASS | `included_in_metrics_eoms = 0`. |

## 3. Quantitative Progress
- **Total Trades (Rows)**: 4,161
- **Total Signals**: ~2,080
- **Unique Configs with Signals**: 19
- **Months Processed**: 2020-01 to 2021-07 (Partial).

## 4. Red Flags & Diagnosis
> [!WARNING]
> **Account Survival**: 19/19 configurations have already hit FTMO blown state (10% Drawdown) before finishing 2021.
> **Edge Observation**: A pesar de que el motor es íntegro, la traducción actual de la estrategia (Quality + Displacement) está demostrando una fragilidad extrema ante los costos de comisión y slippage en el periodo 2020-2021.
> **EOM Ratio**: 100% de los trades son marcados como `artificial_eom`. Esto indica que ninguna posición ha tocado TP o SL antes del cierre de sesión de las 16:55 NY en este periodo inicial.

## 5. Veredicto
**Estado**: `CONTINUE_CLEAN`

**Razón**: Aunque los resultados preliminares son negativos (RED en performance), la integridad del proceso es del 100%. Se debe permitir que la corrida termine al menos la fase TRAIN completa para obtener una diagnóstico estadístico final antes de declarar el `RED_SEALED`. No se recomienda detener prematuramente hasta tener el reporte de TRAIN completo generado por el propio runner.
