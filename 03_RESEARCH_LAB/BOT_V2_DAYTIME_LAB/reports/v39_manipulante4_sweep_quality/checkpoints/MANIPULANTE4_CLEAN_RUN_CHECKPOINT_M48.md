# MANIPULANTE 4 — CLEAN RUN CHECKPOINT [48/76] (End of VAL)

## 1. Integrity Status
- **Status**: `CONTINUE_CLEAN` (to completion)
- **Runner SHA256**: `60e78830f812183ecb12ce7b6a8743c73669f1e5d88176d0745d1ca7ca6aa92f`
- **Runner Frozen**: SÍ.
- **Audit Date**: 2026-05-13 12:41 NY
- **Dataset Phase**: VAL Completed. TEST In progress (Month 49/76).

## 2. Constraint Verification
| Constraint | Status | Notes |
| :--- | :--- | :--- |
| EURUSD Only | PASS | |
| 07:00-17:00 NY | **SOFT_PASS** | 102 trades exited at 17:00:00 NY. technically on the limit. |
| Max 3 trades/day | PASS | 0 violations. |
| News Fail-Close | PASS | |
| Rollover Blocked | PASS | |
| Slippage 0.2 | PASS | |
| Comisión FTMO | PASS | |
| EOM Excluded | PASS | |

## 3. Quantitative Progress
- **Total Trades (Rows)**: 15,307
- **Total Signals**: ~7,653
- **Unique Configs with Signals**: 19
- **VAL Phase Statistics**:
    - Blown accounts: 19/19.

## 4. Red Flags & Diagnosis
> [!CAUTION]
> **Extinción Total**: Al cierre de VAL, el 100% de las configuraciones han fallado. No existe ninguna candidata que haya sobrevivido a la fricción institucional con los filtros de "Quality + Displacement" en el periodo 2020-2023.
> **Veredicto Anticipado**: La investigación M4 se cerrará como **RED** al finalizar la corrida de TEST. No hay soporte para expansión bajo estos parámetros.

## 5. Veredicto
**Estado**: `CONTINUE_CLEAN`

**Razón**: Se permite terminar el 100% de la corrida (fase TEST) para tener el mapa completo del fallo y asegurar que no haya ningún sesgo de selección tardío. El proceso sigue siendo íntegro y auditable.
