# NEXT PROMPT — AUDIT BO01 BACKTEST FRAMEWORK ENTRY POLICY PATCH V1

Actuá como auditor institucional destructivo read-only, Senior Quant Reviewer, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente el patch de política de entrada aplicado al diseño del framework de backtest BO01.

El objetivo NO es ejecutar backtest.
El objetivo NO es cargar datos reales.
El objetivo NO es probar edge.
El objetivo NO es probar rentabilidad.

El objetivo es verificar si el patch:

1. Fue creado solo dentro del scope de documentación autorizado.
2. Establece strictly **`ENTRY_NEXT_CANDLE_OPEN`** como la única política de entrada activa en `BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_V1.md`.
3. Elimina completamente cualquier alternativa de entrada tipo breakout, contract boundary o intrabar.
4. Hardea adecuadamente las abort conditions en el punto 4 de la sección 8 de la especificación de diseño.
5. Neutraliza claims y lenguaje absoluto en el reporte de diseño.
6. No contiene código ejecutable de backtest.
7. No accede a validation, holdout ni años 2025/2026.
8. No busca parámetros ni optimiza.
9. No declara edge ni rentabilidad.

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL PATCH DE POLÍTICA DE ENTRADA BO01, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

============================================================
CONTEXTO A AUDITAR
============================================================

Branch de patch:

research/bo01-backtest-framework-entry-policy-patch-v1-20260518

Archivos a auditar en `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`:

1.
BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_V1.md

2.
BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_REPORT_V1.md

3.
BO01_BACKTEST_FRAMEWORK_ENTRY_POLICY_PATCH_REPORT_V1.md

============================================================
REGLAS ABSOLUTAS DE AUDITORÍA
============================================================

PROHIBIDO:
- NO ejecutar Python.
- NO ejecutar backtest.
- NO cargar datos de mercado.
- NO modificar código, tests, datos ni runner.
- NO git add .
- NO force push, reset --hard, rebase, clean o stash.
- NO declarar edge.
- NO declarar rentabilidad.

PERMITIDO:
- git status/diff/log/show/ls-files/check-ignore.
- rg textual / Select-String.
- Leer markdowns de diseño y patch.
- Crear reporte de auditoría markdown.
- Crear prompt futuro.
- Commit/push de reportes de auditoría autorizados.
