# COST HARDENING PROTOCOL
**Versión**: 1.0
**Estado**: **DRAFT_FOR_TRAIN_ONLY**
**Fecha**: 2026-05-15

## 1. Objective
Endurecer la evidencia de entrenamiento (train-only) aplicando escenarios de estrés de costos realistas para cuantificar la fragilidad de las familias F06, F08 y F12 antes de cualquier fase de validación (V50C).

## 2. Current Risk Profile
- **Slippage**: Actualmente 0.0 pips (Irreal).
- **Comisiones**: Simplificadas (Potencialmente bajas).
- **Estado**: La ventaja detectada puede ser una ilusión estadística causada por la ausencia de fricción de mercado.

## 3. Scope
- **Familias**: F06, F08, F12 únicamente.
- **Periodo**: Mismos 5 meses de Train (2020-2024).
- **Aislamiento**: Prohibido tocar 2025-2026 (Holdout).
- **Criterio**: No se busca optimizar, sino medir la degradación.

## 4. Cost Scenarios (Stress Tests)
| Scenario | Slippage (Pips) | Commission | Notes |
| :--- | :--- | :--- | :--- |
| **BASELINE** | 0.0 | Actual | Referencia de Rerun 68fa2280 |
| **SLIPPAGE_05** | 0.5 round-trip | Actual | Estrés leve (Standard broker) |
| **SLIPPAGE_10** | 1.0 round-trip | Actual | Estrés moderado (High slippage) |
| **FTMO_COST** | 0.5 round-trip | $7/lot | Escenario Institucional/Fondeo |
| **STRESS_COMBO** | 1.0 round-trip | $10/lot | Escenario de "Peor Caso" |

## 5. Pass/Fail Criteria for Validation (V50C)
Una configuración SOLO es apta para Validation si en el escenario **FTMO_COST**:
- **PF Stress** >= 1.20.
- **Net R Stress** > 0.
- **Avg Trade** >= 1.5 pips (para cubrir costos y ruido).
- **Degradación de PF** <= 40% vs Baseline.

## 6. Decision Decision
- **COST_ROBUST**: Lista para diseño de V50C.
- **COST_FRAGILE**: Requiere re-diseño de filtros o gestión.
- **COST_REJECTED**: Ventaja inexistente bajo fricción real.

## 7. Execution Note
Este protocolo se ejecuta estrictamente sobre los resultados ya certificados del RunID 68fa2280.
