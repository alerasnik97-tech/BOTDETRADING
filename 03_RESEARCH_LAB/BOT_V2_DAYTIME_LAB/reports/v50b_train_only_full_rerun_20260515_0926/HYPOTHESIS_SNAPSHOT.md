# HYPOTHESIS SNAPSHOT - V50B TRAIN-ONLY FULL RERUN
**Fecha**: 2026-05-15
**RunID Previsto**: Dinámico (Single-Writer Isolated)

## Objetivo del Rerun
Obtener evidencia cuantitativa limpia y auditable sobre el rendimiento de las familias F06, F08 y F12 utilizando únicamente el set de Entrenamiento (TRAIN). Este rerun invalida cualquier resultado previo contaminado por el incidente Multi-RunID.

## Familias Evaluadas
1. **F06 (Volatility Regime)**: Hipótesis de ventaja en expansión de volatilidad tras consolidación. Mostró señales prometedoras en micro-smoke 2022-05.
2. **F08 (Session Overlap)**: Hipótesis de explotación de liquidez en el solapamiento London/NY.
3. **F12 (Macro Safe Window)**: Hipótesis de filtrado por RSI en ventanas de baja volatilidad macro.

## Meses Train Seleccionados (EURUSD)
- **2020-03** (Alta volatilidad, COVID crash)
- **2021-08** (Rango estacional)
- **2022-05** (Micro smoke baseline)
- **2023-01** (Inicio de ciclo post-inflación)
- **2024-04** (Reciente, régimen actual)

## Restricciones y Seguridad
- **Sesión**: 07:00 - 17:00 NY.
- **Max Trades/Day**: 3.
- **Holdout/Test**: BLOQUEADO (2025-2026 intocable).
- **Single-Writer**: Obligatorio con Lock Atómico.

## Criterios de Éxito Preliminar
- **Rechazo**: PF < 1.0 o Net R negativo en el agregado de los 5 meses.
- **Watchlist**: PF 1.0 - 1.2.
- **Promising**: PF > 1.2 con estabilidad mensual (mínimo 3/5 meses positivos).

## Declaración de Intenciones
Este rerun busca **evidencia técnica**, no validación de estrategia para fondeo. No se está probando el borde (edge) final, sino la capacidad del motor para producir señales reales en datos históricos reales sin sesgos de concurrencia.
