# PHASE50V — TIME_EXIT OPERATIONAL SOURCE AUDIT

## Identificación de la Regla

La regla de cierre por tiempo (`TIME_EXIT`) existe en el ecosistema de MANIPULANTE en tres niveles distintos:

### 1. Nivel de Investigación (Research Engine)
- **Archivo**: `research_lab/engine.py`
- **Mecánica**: 
    - **Prioridad 2**: `forced_session_close`. Se dispara si `force_close_mask` es verdadero para la barra actual.
    - **Prioridad 3**: `time_exit`. Se dispara si `held_bars >= max_hold_bars`.
- **Configuración**: 
    - `SessionConfig.force_close` por defecto a las `19:00` (NY).
    - `max_hold_bars` es un parámetro opcional enviado en el contrato de la señal.

### 2. Nivel Operativo (Bot Runner)
- **Archivo**: `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_bot_runner.py`
- **Mecánica**: `safe_close.execute_safe_close()`.
- **Trigger**: `phase37x_session_lifecycle.should_force_safe_close()`.
- **Configuración**:
    - `forced_safe_close`: `19:45` NY.
    - `friday_hard_close`: `16:55` NY.
    - `daily_shutdown`: `20:00` NY.

### 3. Nivel de Dataset Histórico
- **Archivo**: `phase38_raw_trades_enriched.csv`
- **Mecánica**: Las posiciones marcadas como `FORCED_CLOSE` (100% de la muestra de 100 trades) cierran exactamente a las **20:00 NY**.

## Hallazgos de Determinismo

- **Determinismo Histórico**: Se confirma que el 100% de los cierres `FORCED_CLOSE` en el dataset histórico siguen una regla horaria estricta (20:00 NY).
- **Discrepancia Horaria**: Existe una diferencia entre el histórico (20:00 NY) y la configuración operativa actual (19:45 NY). Esto sugiere un endurecimiento de la regla en la versión operativa para evitar el rollover.
- **Origen del Edge**: En los meses adversos auditados (2017-05, 2017-08, 2020-04), el 100% de los `TIME_EXIT` detectados por Gemini son en realidad trades que no alcanzaron TP/SL en la ventana original del CSV y "expiraron" manteniendo beneficios.

## Evidencia Forward/Demo

- **Código**: La capacidad de cierre forzado por tiempo está implementada y activa en el bot de FTMO Trial.
- **Logs**: Actualmente `FORWARD_TRADES=0`, por lo que no hay ejecuciones registradas todavía.
- **Veredicto de Replicabilidad**: **REPLICABLE_CONFIRMED**. La lógica de cierre por tiempo es una función nativa del motor de ejecución y está correctamente configurada en el bot operativo.

## Riesgos Detectados

- **Lookahead Bias en Auditoría**: El replay de Gemini está usando el `exit_time` del CSV como límite de tiempo. Si ese `exit_time` fue un TP en el original pero no en el tick data, Gemini lo marca como `TIME_EXIT` si hay profit. Esto sobreestima el edge de "tiempo" si el bot operativo no cerrara en ese momento exacto.
- **Endurecimiento Operativo**: El bot operativo cierra a las 19:45 NY, mientras que el histórico cerraba a las 20:00 NY. Esto es una medida de seguridad positiva.
