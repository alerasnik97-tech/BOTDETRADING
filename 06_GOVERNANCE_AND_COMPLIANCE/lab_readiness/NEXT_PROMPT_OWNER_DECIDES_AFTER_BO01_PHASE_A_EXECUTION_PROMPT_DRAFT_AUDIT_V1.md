# NEXT PROMPT — OWNER DECIDES AFTER BO01 PHASE A EXECUTION PROMPT DRAFT AUDIT V1

Actuá como Senior Quant Architect, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Presentar al owner las opciones de continuación tras el pase con warnings de la auditoría externa del borrador del prompt de ejecución Phase A.

ESTA FASE NO IMPLEMENTA CÓDIGO, NO CARGA DATOS REALES Y NO CORRE PYTHON.

============================================================
LÍMITES DE SEGURIDAD ESTRICTOS
============================================================

- NO acceder a datos de validation ni holdout.
- NO procesar datos de los años 2025 o 2026.
- NO realizar sweeps de optimización ni búsquedas de parámetros.
- NO realizar backtests ni simulaciones sobre datos reales en esta fase de decisión.
- NO autorizar ni habilitar cuentas demo, real, FTMO ni entornos productivos.

---

## OPCIONES DE DECISIÓN PARA EL OWNER

El owner debe seleccionar una única opción para la siguiente fase:

### OPCIÓN A — EJECUTAR PHASE A BO01 TRAIN-ONLY REAL-DATA BACKTEST
- **Objetivo**: Autorizar y ejecutar la simulación de Phase A (ventana acotada de 5 días de 2015-01-05 a 2015-01-09) en una rama separada para verificar la fontanería real del backtesting.
- **Nota**: Se recomienda revisar los warnings menores (W-01, W-02, W-03) en el informe de auditoría antes de proceder.
- **Datos**: Solo la ventana autorizada de M5 real train-only.

### OPCIÓN B — PARCHEAR WARNINGS DE LA AUDITORÍA
- **Objetivo**: Corregir los tres warnings menores identificados en la auditoría (link de commit del runner, redacción del script temporal y ambigüedad de `train_run` en el handoff) antes de proceder.

### OPCIÓN C — PAUSAR / FREEZE
- **Objetivo**: Congelar la actividad.

---

## DECLARACIÓN REQUERIDA DEL OWNER

El owner deberá responder con una de las siguientes autorizaciones exactas para desbloquear el siguiente prompt:

**Para Opción A**:
“AUTORIZO EJECUTAR PHASE A BO01 TRAIN-ONLY REAL-DATA BACKTEST, VENTANA 2015-01-05 A 2015-01-09, SOLO TRAIN-ONLY, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN OPTIMIZATION/SWEEP, SIN DEMO/REAL/FTMO Y SIN EDGE CLAIMS.”

**Para Opción B**:
“AUTORIZO PARCHEAR WARNINGS DE LA AUDITORÍA DEL PROMPT DE EJECUCIÓN PHASE A, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

**Para Opción C**:
“AUTORIZO CONGELAR TEMPORALMENTE LA ACTIVIDAD DE BACKTESTING BO01, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”
