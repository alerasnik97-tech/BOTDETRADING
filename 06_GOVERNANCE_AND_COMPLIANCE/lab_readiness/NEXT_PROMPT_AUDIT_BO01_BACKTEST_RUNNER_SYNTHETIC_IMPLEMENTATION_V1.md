# NEXT PROMPT — AUDIT BO01 BACKTEST RUNNER SYNTHETIC IMPLEMENTATION V1

Actuá como auditor institucional destructivo read-only, Senior Python Reviewer, Quant Infrastructure Auditor, Backtesting Auditor y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente la implementación del primer runner de backtesting para la estrategia BO01 (`bo01_backtest_runner.py`) y sus tests unitarios sintéticos.

El objetivo NO es ejecutar backtests.
El objetivo NO es cargar datos de mercado.
El objetivo NO es optimizar parámetros.
El objetivo NO es declarar rentabilidad.

El objetivo único es verificar si la infraestructura implementada cumple rigurosamente con:

1. **File Scope**: Solo se crearon los archivos autorizados en el whitelist. No se alteró ningún archivo del core, estrategias (`BO01Strategy.py`), loaders o base de datos.
2. **Runner Pureza**: `bo01_backtest_runner.py` no realiza lectura/escritura de archivos (`read_csv`, `to_csv`, `open(`, `Path(`) ni importa secretos, configuraciones o bases de datos reales.
3. **Guardas de Fechas y Particiones**: El runner contiene guardas explícitas y activas para fallar cerrado si el frame contiene fechas en 2025 o 2026, o si las columnas de split contienen etiquetas como 'validation' o 'holdout'.
4. **Política de Entrada Estricta**: La política `ENTRY_NEXT_CANDLE_OPEN` está implementada de manera pura: entrada en el precio `Open` de la vela $t+1$ posterior a la confirmación de la señal en la vela $t$. No existen filtros intrabar ni de breakout activos.
5. **Gestión de Posiciones**: Límite estricto de máximo 1 trade activo por vez y máximo 1 trade por día (primera señal válida del día), ignorando señales posteriores en el mismo día.
6. **Resolución de Salidas y Costos**: Salidas resueltas cronológicamente vela a vela con política conservadora `STOP_FIRST` en misma barra. Deducción determinística de costos (spread, slippage y comisión round-turn USD convertida a R).
7. **Test Coverage Sintético**: Verificación de que los 21 tests de contrato, ejecución y seguridad son puramente sintéticos, no tocan disco, y pasan con 100% de éxito.
8. **Git Safety**: Confirmación de que el branch es correcto y no se commitearon outputs locales ni dirty preexistente.

============================================================
LÍMITES DE SEGURIDAD ESTRICTOS
============================================================

- NO ejecutar Python sobre archivos de datos reales.
- NO cargar archivos de `05_MARKET_DATA_VAULT`.
- NO acceder a particiones de validation ni holdout.
- NO procesar datos de los años 2025 o 2026.
- NO realizar sweeps de optimización ni búsquedas de parámetros.
- NO autorizar ni habilitar cuentas demo, real, FTMO ni entornos productivos.

---

## ESTRUCTURA DEL INFORME DE AUDITORÍA REQUERIDO

El auditor deberá generar un informe detallado en `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/` bajo el nombre:
`BO01_BACKTEST_RUNNER_SYNTHETIC_IMPLEMENTATION_EXTERNAL_AUDIT_V1.md`

Debe clasificar cada verificación con:
`PASS` / `WARN` / `BLOCKER`

Y concluir con un estado final claro:
`BO01_BACKTEST_RUNNER_SYNTHETIC_IMPLEMENTATION_AUDIT_PASS` o blocker.
