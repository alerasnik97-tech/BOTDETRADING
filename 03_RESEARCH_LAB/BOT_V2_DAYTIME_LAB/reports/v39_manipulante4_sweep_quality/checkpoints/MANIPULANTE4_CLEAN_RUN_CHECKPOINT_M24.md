# MANIPULANTE 4 — CLEAN RUN CHECKPOINT [24/76] (End of TRAIN)

## 1. Integrity Status
- **Status**: `CONTINUE_CLEAN`
- **Runner SHA256**: `60e78830f812183ecb12ce7b6a8743c73669f1e5d88176d0745d1ca7ca6aa92f`
- **Runner Frozen**: SÍ.
- **Audit Date**: 2026-05-13 12:16 NY
- **Dataset Phase**: TRAIN Completed. VAL In progress (Month 25/76).

## 2. Constraint Verification
| Constraint | Status | Notes |
| :--- | :--- | :--- |
| EURUSD Only | PASS | |
| 07:00-17:00 NY | PASS | |
| Max 3 trades/day | PASS | 0 violations. |
| News Fail-Close | PASS | |
| Rollover Blocked | PASS | |
| Slippage 0.2 | PASS | |
| Comisión FTMO | PASS | |
| EOM Excluded | PASS | |

## 3. Quantitative Progress
- **Total Trades (Rows)**: 4,555
- **Total Signals**: ~2,277
- **Unique Configs with Signals**: 19
- **TRAIN Phase Statistics**:
    - Months with 0 trades: 2021-08 to 2021-12.
    - Blown accounts: 19/19.

## 4. Red Flags & Diagnosis
> [!CAUTION]
> **Total Failure in TRAIN**: Todas las configuraciones (19) han quemado la cuenta de $100k antes de finalizar la fase de entrenamiento.
> **Kill Criteria**: El PF de TRAIN será < 1.0 para todas las configuraciones.
> **Protocolo**: Siguiendo las reglas de "Agilidad Controlada", la corrida debe continuar hasta generar los archivos de resultados oficiales de TRAIN para sellar la investigación como RED. Sin embargo, ya es evidente que el desplazamiento exigido y la calidad de barrido no son suficientes para compensar la fricción institucional en este conjunto de parámetros.

## 5. Veredicto
**Estado**: `CONTINUE_CLEAN` (to formalize RED closure)

**Razón**: La integridad es perfecta. Se permite completar VAL (que ya ha empezado con `394t` en Enero 2022) para ver si existe algún cambio de régimen que salve alguna configuración, aunque la probabilidad es baja dado el estado de TRAIN.
