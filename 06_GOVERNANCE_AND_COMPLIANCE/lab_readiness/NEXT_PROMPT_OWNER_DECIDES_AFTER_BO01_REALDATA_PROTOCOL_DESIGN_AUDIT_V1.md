# NEXT PROMPT — OWNER DECIDES AFTER BO01 REAL-DATA PROTOCOL DESIGN AUDIT V1

Actuá como Senior Quant Architect, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Presentar al owner las opciones de continuación tras el pase de la auditoría externa del diseño del protocolo de backtest BO01 train-only real-data.

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

### OPCIÓN A — DISEÑAR EL PROMPT DE EJECUCIÓN PHASE A BO01 TRAIN-ONLY REAL-DATA
- **Objetivo**: Diseñar el prompt técnico detallado para proceder a la ejecución de Phase A (Plumbing Smoke Backtest, ventana acotada de 5 días de 2015-01-05 a 2015-01-09) en una rama separada.
- **Justificación**: Especificar con rigor absoluto el código del cargador, las assertions del data proof, la parametrización de los costos y la captura de logs para asegurar una ejecución 100% auditable.
- **Datos**: Ninguno en esta fase (solo diseño de prompt escrito).

### OPCIÓN B — PAUSAR / FREEZE
- **Objetivo**: Congelar la actividad de backtesting.

---

## DECLARACIÓN REQUERIDA DEL OWNER

El owner deberá responder con una de las siguientes autorizaciones exactas para desbloquear el siguiente prompt:

**Para Opción A**:
“AUTORIZO DISEÑAR EL PROMPT DE EJECUCIÓN PHASE A BO01 TRAIN-ONLY REAL-DATA, SIN EJECUTAR BACKTEST CON DATOS REALES, SIN CARGAR DATOS DE MERCADO, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

**Para Opción B**:
“AUTORIZO CONGELAR TEMPORALMENTE LA ACTIVIDAD DE BACKTESTING BO01, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”
