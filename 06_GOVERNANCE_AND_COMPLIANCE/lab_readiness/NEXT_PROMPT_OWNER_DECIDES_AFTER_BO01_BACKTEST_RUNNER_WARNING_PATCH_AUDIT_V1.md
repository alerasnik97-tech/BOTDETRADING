# NEXT PROMPT — OWNER DECIDES AFTER BO01 BACKTEST RUNNER WARNING PATCH AUDIT V1

Actuá como Senior Quant Architect, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Presentar al owner las opciones de continuación tras el pase de la auditoría externa del BO01 backtest runner con warnings parchados.

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

### OPCIÓN A — DISEÑAR EL PROTOCOLO DE PRIMERA EJECUCIÓN BO01 TRAIN-ONLY REAL-DATA
- **Objetivo**: Diseñar la especificación metodológica estricta para ejecutar el primer backtest de BO01 con datos reales de mercado (M5 train-only, ventana acotada de 2015 a 2024), detallando costos aplicados, ventanas operativas y reconciliaciones.
- **Justificación**: Definir con precisión científica cómo cargaremos y procesaremos los datos de EURUSD M5 train sin incurrir en lookahead ni data leakage, y cómo presentaremos los reportes de performance estructural.
- **Datos**: Ninguno en esta fase (solo diseño metodológico escrito).

### OPCIÓN B — PAUSAR / FREEZE
- **Objetivo**: Congelar la actividad de backtesting.

---

## DECLARACIÓN REQUERIDA DEL OWNER

El owner deberá responder con una de las siguientes autorizaciones exactas para desbloquear el siguiente prompt:

**Para Opción A**:
“AUTORIZO DISEÑAR EL PRIMER PROTOCOLO DE EJECUCIÓN DE BACKTEST BO01 TRAIN-ONLY REAL-DATA, SIN EJECUTAR BACKTEST CON DATOS REALES, SIN CARGAR DATOS DE MERCADO, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

**Para Opción B**:
“AUTORIZO CONGELAR TEMPORALMENTE LA ACTIVIDAD DE BACKTESTING BO01, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”
