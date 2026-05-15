> ---
> ## ⛔ SUPERSEDED / NOT CERTIFIED — DO NOT USE (Addendum 2026-05-15)
> **This report is superseded and must not be used for promotion/validation
> decisions until the evidence rebuild is complete.**
> Input real = archivo cuarentenado 91.7% contaminado; F06 "COST_ROBUST"
> sobre N=10; sin modelo de spread (STRESS_COMBO no es worst-case). Ver
> `reports/pre_claude_blocker_remediation/SUPERSEDED_CERTIFICATIONS.md` y
> `FORENSIC_VERIFICATION_REPORT.md`. Contenido original conservado abajo
> intacto como evidencia del incidente.
> ---

# V50B TRAIN-ONLY COST HARDENING REPORT
**Status**: **COMPLETED_AUDIT_READY**
**Fecha**: 2026-05-15

## 1. Executive Summary
Se ha ejecutado el protocolo de endurecimiento de costos sobre las familias F06, F08 y F12 utilizando la evidencia certificada del RunID 68fa2280. Los resultados revelan una fragilidad crítica en F08 y F12 frente a la fricción real del mercado, mientras que F06 se consolida como la única familia robusta apta para continuar el proceso institucional.

## 2. Context
- **Canonical RunID Base**: 68fa2280
- **Source of Truth Base**: MASTER_RANKING.csv / TRADES.csv
- **Cost Model Previous Status**: RESEARCH_ONLY_ZERO_SLIPPAGE
- **Isolation Status**: 100% Train-only (No test/holdout touched).

## 3. Git Context
- **Branch**: research/v50b-cost-hardening-foundation-20260515
- **Commit SHA**: a7675478 (foundation)
- **Lineage Status**: Repaired (Compatible con PR).

## 4. Cost Scenarios
| Scenario | Slippage | Commission | Status |
| :--- | :--- | :--- | :--- |
| BASELINE | 0.0 | $0/lot | Reference |
| SLIPPAGE_05 | 0.5 pips | $0/lot | Stress Test |
| SLIPPAGE_10 | 1.0 pips | $0/lot | Stress Test |
| **FTMO_COST** | 0.5 pips | $7/lot | **Standard Gate** |
| STRESS_COMBO | 1.0 pips | $10/lot | Worst Case |

## 5. Safety Verification
- **test_touched**: NO
- **validation_touched**: NO (Scope limitado a Train-only)
- **holdout_touched**: NO
- **raw_data_mutated**: NO
- **sweep_run**: NO
- **optimization_run**: NO
- **2025_2026_touched**: NO

## 6. Results by Family (Aggregated Stress)
| Family | BASELINE PF | FTMO_COST PF | Net R Stress (Avg) | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **F06** | 2.20 | **1.49** | +3.04 | **COST_ROBUST** |
| **F08** | 1.00 | 0.57 | -1.95 | **COST_REJECTED** |
| **F12** | 0.77 | 0.66 | -4.18 | **COST_REJECTED** |

## 7. Fragility Analysis
- **F06**: Muestra una degradación controlada. Con 1.0 pips de slippage (SLIPPAGE_10) el PF baja a 1.59, pero se mantiene rentable. En el escenario FTMO_COST (0.5 pips + $7/lot), el PF de 1.49 es institucionalmente aceptable.
- **F08/F12**: Su ventaja estadística desaparece por completo ante cualquier fricción. Esto sugiere que sus "bordes" (edges) eran ilusorios o dependían de movimientos demasiado pequeños para ser capturados con costos reales.

## 8. Certification Decision
**CERTIFIED_FOR_F06_ONLY**. Se aprueba únicamente a la familia F06 para avanzar al diseño del V50C Validation Plan. F08 y F12 quedan descartadas de la línea principal de producción por fragilidad de costos.

## 9. Next Recommended Step
**CLAUDE_EXTREME_AUDIT (F06 Only)**: Someter a F06 a una auditoría forense de lógica antes de tocar cualquier dato de 2025.

## 10. Copy-Paste Summary for ChatGPT
- **Status**: SUCCESS (F06 Robust | F08/F12 Rejected)
- **Method**: Cost Hardening (Slippage Stress 0.5/1.0)
- **Source of Truth**: 68fa2280 certified trades.
- **Next Step**: F06 Audit / V50C Planning (Deferred).
