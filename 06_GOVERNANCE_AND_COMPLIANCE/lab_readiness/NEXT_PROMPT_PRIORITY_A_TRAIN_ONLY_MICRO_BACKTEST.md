# NEXT PROMPT: EURUSD Priority A Train-Only Micro-Backtest

Actúa como **Senior Quant Trader & Research Operator**. Tu misión es ejecutar una micro-corrida de backtest controlada para las estrategias Priority A aprobadas por auditoría externa.

## Objetivo
Verificar el performance preliminar de las 4 estrategias Priority A en datos de entrenamiento (2015-2024), una por una, asegurando la integridad de los outputs.

## Alcance
- **Dataset**: EURUSD M1 (o M5 según estrategia) de entrenamiento únicamente.
- **Ventana**: 2015-01-01 a 2024-12-31.
- **Estrategias**:
  1. `mr01_anchor_elastic`
  2. `mr02_vwap_stretch_reversion`
  3. `tp01_london_ny_momentum_pullback`
  4. `ve_orb_volatility_expansion`

## Protocolo de Ejecución
1. **Configuración**: Usar `EngineConfig` estándar con costos realistas (Spread 0.8-1.2 pips, Slippage 0.2-0.5 pips).
2. **Serialización**: Ejecutar una estrategia a la vez. NO lanzar procesos paralelos masivos.
3. **No Optimization**: Usar los `DEFAULT_PARAMS` definidos en cada archivo. No realizar sweeps ni búsquedas de parámetros en esta fase.
4. **No Leakage**: Validar vía `lab_preflight` que no se toquen datos de 2025/2026.
5. **Output**: Generar reportes en `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/priority_a_micro_run/`.

## Restricciones Inviolables
- **No Holdout**: Prohibido el acceso a la carpeta `sealed_holdout` o fechas >= 2025.
- **No Validation**: No realizar validación cruzada ni WFA todavía.
- **No Production**: El código debe permanecer en ramas de `research`.
- **Atomic Writes**: Asegurar que cada corrida tenga un `run_id` único para evitar contaminación de CSVs.

## Tareas
- Correr micro-backtest para cada una de las 4 estrategias.
- Recopilar métricas base (PF, Net PnL, Drawdown, N trades).
- Generar curvas de equidad preliminares.
- Documentar si alguna estrategia emite errores en tiempo de ejecución (Run-time Audit).
