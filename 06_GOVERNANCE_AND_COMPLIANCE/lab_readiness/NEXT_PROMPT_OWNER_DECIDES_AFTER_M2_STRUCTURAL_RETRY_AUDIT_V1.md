# NEXT PROMPT — OWNER DECIDES AFTER M2 STRUCTURAL RETRY AUDIT V1

Actuá como Senior Python Engineer, Quant Infrastructure Auditor y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Presentar al owner las opciones de continuación del laboratorio cuantitativo tras la aprobación de la auditoría externa M2 Conservative Structural Retry.

ESTA FASE NO EJECUTA NADA, NO CARGA DATOS REALES Y NO HACE BACKTEST.

============================================================
LÍMITES DE SEGURIDAD ESTRICTOS
============================================================

- NO acceder a datos de validation ni holdout.
- NO procesar datos de los años 2025 o 2026.
- NO realizar sweeps de optimización ni búsquedas de parámetros.
- NO realizar backtest formales ni simulaciones de PnL en esta fase de decisión.
- NO autorizar ni habilitar cuentas demo, real, FTMO ni entornos productivos.

---

## OPCIONES DE DECISIÓN PARA EL OWNER

El owner debe seleccionar una única opción para la siguiente fase:

### OPCIÓN A — EXPANDIR EVALUACIÓN ESTRUCTURAL A 12 MESES
- **Objetivo**: Correr el M2 Structural Runner sobre la totalidad del año 2015 en el dataset de train.
- **Justificación**: Probar el comportamiento del runner ante un volumen de datos 4 veces mayor (~72,000 velas M5) para confirmar la estabilidad de los conteos y recolectar estadísticas mensuales completas de estacionalidad sin calcular performance.
- **Datos**: EURUSD prepared train 2015-2024.
- **Rango**: 2015-01-01 00:00:00 UTC a 2015-12-31 23:59:59 UTC.

### OPCIÓN B — DISEÑAR EL PRIMER BACKTEST TRAIN-ONLY PARA BO01
- **Objetivo**: Diseñar la especificación del primer backtest controlado y liviano en train-only para la estrategia BO01 (London Breakout), manteniendo a MR02 en observación estricta por su extremadamente baja frecuencia estructural (5 señales en 3 meses).
- **Justificación**: Definir la lógica de simulación de ejecuciones, perfil de costos (base/conservative/stress), spread y deslizamiento de manera segura, sin tocar la rama de producción y sin optimizar.
- **Datos**: EURUSD prepared train 2015-2024.
- **Rango**: 2015-01-05 a 2015-01-09 (ventana de prueba plumbing).

### OPCIÓN C — PAUSAR Y ENMEDAR DOCUMENTACIÓN DE AUDITORÍA
- **Objetivo**: Parchear las inconsistencias documentales menores detectadas en la auditoría externa de ejecución (W-01 y W-02) antes de proceder a cualquier otra fase operativa o de backtesting.
- **Justificación**: Mantener la higiene institucional extrema en la documentación del laboratorio de investigación.

---

## DECLARACIÓN REQUERIDA DEL OWNER

El owner deberá responder con una de las siguientes autorizaciones exactas para desbloquear el siguiente prompt:

**Para Opción A**:
“AUTORIZO DISEÑAR E IMPLEMENTAR LA EXPANSIÓN ESTRUCTURAL M2 A 12 MESES EN TRAIN-ONLY, SIN EJECUTAR M2, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

**Para Opción B**:
“AUTORIZO DISEÑAR EL PRIMER FRAMEWORK DE BACKTEST TRAIN-ONLY PARA BO01, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

**Para Opción C**:
“AUTORIZO CORREGIR APLICANDO PARCHES DOCUMENTALES MENORES DE AUDITORÍA M2, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”
