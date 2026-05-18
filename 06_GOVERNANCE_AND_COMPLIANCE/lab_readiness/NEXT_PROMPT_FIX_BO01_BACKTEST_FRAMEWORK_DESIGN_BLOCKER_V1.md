# NEXT PROMPT — FIX BO01 BACKTEST FRAMEWORK DESIGN BLOCKER V1

Actuá como Senior Quant Backtesting Architect y Risk Governance Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Corregir el blocker detectado en la auditoría externa de diseño de backtest BO01:

**`AUDIT_BLOCKED_ENTRY_POLICY_AMBIGUOUS`**

El objetivo único es aplicar un patch documental menor en `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_V1.md` para eliminar la ambigüedad en la política de entrada y establecer strictly **`ENTRY_NEXT_CANDLE_OPEN`** como la única política autorizada.

ESTA FASE NO IMPLEMENTA CÓDIGO, NO EJECUTA PYTHON Y NO CARGA DATOS DE MERCADO.

============================================================
LÍMITES DE SEGURIDAD ESTRICTOS
============================================================

- NO acceder a datos de validation ni holdout.
- NO procesar datos de los años 2025 o 2026.
- NO realizar sweeps de optimización ni búsquedas de parámetros.
- NO realizar backtest ni simulaciones de PnL.
- NO autorizar ni habilitar cuentas demo, real, FTMO ni entornos productivos.

---

## REQUERIMIENTOS DEL PATCH DOCUMENTAL

El patch debe modificar la sección **"4. Execution Model"** punto 5 de:
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_V1.md`

Debe quedar de la siguiente manera:
1. **Entrada Única**: Declarar explícitamente que la entrada ocurre **única y estrictamente al precio Open de la siguiente vela ($t+1$)** tras confirmarse la señal al cierre de la vela $t$.
2. **Eliminar Ambigüedades**: Eliminar cualquier referencia alternativa de entrada (como breakout del breakout price o boundaries).
3. **Justificación Técnica**: Explicar que esta política garantiza determinismo absoluto a nivel de barra, es 100% causal, reproducible, simplifica el modelo intrabar y elimina riesgos de lookahead bias al no requerir subdivisión temporal.

---

## DECLARACIÓN REQUERIDA DEL OWNER

El owner deberá responder con la siguiente autorización exacta para desbloquear el siguiente prompt:

“AUTORIZO PARCHEAR LA POLÍTICA DE ENTRADA BO01 PARA FIJAR STRICTLY ENTRY_NEXT_CANDLE_OPEN EN EL DISEÑO DEL FRAMEWORK DE BACKTEST, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”
