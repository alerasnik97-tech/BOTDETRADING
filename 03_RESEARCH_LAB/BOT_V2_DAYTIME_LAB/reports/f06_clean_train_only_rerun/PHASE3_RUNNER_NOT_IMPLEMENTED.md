# PHASE 3 RUNNER NOT IMPLEMENTED

## 1. Status
**RUNNER_NOT_IMPLEMENTED**

## 2. Assessment
El pipeline actual (`f06_rebuild_pipeline.py`) funciona exclusivamente como un andamio institucional (`scaffold`). Los únicos comandos que expone son:
- `validate_config`
- `dry_run`
- `validate_outputs`

Según el diseño documentado, el script declara explícitamente:
> `F06 evidence rebuild scaffold (NO strategy / NO backtest).`

## 3. Decision
Siguiendo las reglas absolutas de esta FASE 3, **NO se ha improvisado un runner real**, y **NO se han utilizado scripts de legacy V50B** (que están fuertemente contaminados o cuarentenados). 

El proyecto carece actualmente del módulo seguro que enlace la nueva `Foundation V2` con la superficie real del engine visible en este checkout (`research_lab/engine.py`) bajo las reglas fail-closed requeridas (por ejemplo, impidiendo inyección de datos 2025/2026 a nivel de runner o aplicando el cálculo de slippage/spread requerido antes de escribir los CSVs de ranking/ledger).

## 4. Next Step
El resultado final de esta operación debe derivar en:
**READY_FOR_PHASE3_RUNNER_IMPLEMENTATION**

Es necesario desarrollar el comando de ejecución real (ej. `run_strategy`) que respete estrictamente los schemas exigidos en esta FASE 3.
