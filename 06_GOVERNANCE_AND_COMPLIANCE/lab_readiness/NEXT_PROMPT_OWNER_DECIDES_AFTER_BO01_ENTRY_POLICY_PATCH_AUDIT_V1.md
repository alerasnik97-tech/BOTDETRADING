# NEXT PROMPT — OWNER DECIDES AFTER BO01 ENTRY POLICY PATCH AUDIT V1

Actuá como Senior Python Engineer, Quant Research Infrastructure Engineer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Presentar al owner las opciones de continuación tras la aprobación de la auditoría externa del patch de política de entrada del backtest BO01.

ESTA FASE NO IMPLEMENTA CÓDIGO, NO CARGA DATOS REALES Y NO CORRE PYTHON.

============================================================
LÍMITES DE SEGURIDAD ESTRICTOS
============================================================

- NO acceder a datos de validation ni holdout.
- NO procesar datos de los años 2025 o 2026.
- NO realizar sweeps de optimización ni búsquedas de parámetros.
- NO realizar backtest ni simulaciones de PnL con datos reales en esta fase de decisión.
- NO autorizar ni habilitar cuentas demo, real, FTMO ni entornos productivos.

---

## OPCIONES DE DECISIÓN PARA EL OWNER

El owner debe seleccionar una única opción para la siguiente fase:

### OPCIÓN A — DISEÑAR E IMPLEMENTAR RUNNER DE BACKTEST BO01 CON TESTS SINTÉTICOS
- **Objetivo**: Diseñar e implementar la estructura del primer runner de backtesting para BO01 (`03_RESEARCH_LAB/research_lab/runners/bo01_backtest_runner.py`) y sus correspondientes **tests unitarios sintéticos**, sin cargar datos reales ni ejecutar backtests sobre el dataset de mercado.
- **Justificación**: Validar a nivel de test unitario sintético (con mocks y dataframes de prueba de 5 a 10 filas creados en memoria) que la lógica de step row-by-row, exits chronologically, same-bar stop-first, y deducción de costos estáticos funciona perfectamente a nivel de código sin peligro de lookahead leakage.
- **Datos**: Solo datos sintéticos creados en memoria por los tests.

### OPCIÓN B — PAUSAR Y ENMEDAR ADICIONALMENTE LA DOCUMENTACIÓN DE AUDITORÍA
- **Objetivo**: Aplicar parches documentales cosméticos adicionales para pulir cualquier otra advertencia menor o descripción histórica de blockers en los reportes antes de seguir adelante.
- **Justificación**: Higiene documental extrema del laboratorio quant.

### OPCIÓN C — PAUSAR / FREEZE
- **Objetivo**: Congelar temporalmente la actividad del laboratorio de backtesting para revisión estratégica del portfolio.

---

## DECLARACIÓN REQUERIDA DEL OWNER

El owner deberá responder con una de las siguientes autorizaciones exactas para desbloquear el siguiente prompt:

**Para Opción A**:
“AUTORIZO IMPLEMENTAR EL ESTRUCTURAL BO01 BACKTEST RUNNER CON TESTS SINTÉTICOS Y SIN DATOS REALES, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

**Para Opción B**:
“AUTORIZO CORREGIR APLICANDO PARCHES DOCUMENTALES MENORES DE AUDITORÍA DE ENTRADA BO01, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

**Para Opción C**:
“AUTORIZO CONGELAR TEMPORALMENTE LA ACTIVIDAD DE BACKTESTING BO01, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”
