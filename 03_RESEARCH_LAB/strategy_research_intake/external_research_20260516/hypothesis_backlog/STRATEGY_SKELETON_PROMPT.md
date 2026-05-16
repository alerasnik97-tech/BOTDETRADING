# NEXT PROMPT: EURUSD Strategy Skeleton Implementation (Priority A)

Actúa como **Lead Quant Developer & Engine Architect**. Tu misión es implementar los esqueletos de señal para las estrategias **Priority A** identificadas en el backlog de investigación.

## Objetivo
Traducir las lógicas de las estrategias Priority A a archivos Python estructurados dentro de `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/strategies/`, asegurando compatibilidad total con el motor v7.

## Estrategias a Implementar (Priority A)
1. **Anchor Elastic (MR-01)**
2. **RV Shock Break (VE-01)**
3. **Trend Day EMA Pullback (TP-01)**
4. **Europe Extreme Failure (SD-01)**
5. **Post-News Stabilization (ED-01)**

## Requisitos de Código
- **Standalone**: Cada estrategia debe ser una clase/función independiente que reciba el OHLCV de entrada y parámetros de configuración.
- **Fail-Closed**: Si faltan datos o indicadores, la señal debe ser `NEUTRAL` (0).
- **No Execution**: Este prompt es para lógica de señal y gestión de riesgo básica, NO para ejecución en vivo.
- **Backtest Friendly**: Debe devolver un objeto `Signal` o similar (Long/Short/Flat) junto con los niveles de `Stop Loss` y `Take Profit` calculados según la especificación.

## Protocolo de Implementación
1. Crear una rama nueva: `research/EURUSD-strategy-skeletons-20260516`.
2. Para cada estrategia:
   - Crear el archivo `.py` en la carpeta de estrategias.
   - Implementar los indicadores necesarios (p.ej. VWAP anclado, Donchian, ATR Percentile).
   - Codificar las reglas de entrada exactas (Gatillo + Confirmación).
   - Definir los parámetros por defecto (round numbers) indicados en el backlog.
3. Crear un archivo `STRATEGY_MAP.md` que vincule cada archivo con su ID de backlog.

## Constraints Inviolables
- **Train-Only**: No incluir lógica que cargue datos de 2025/2026.
- **Core Engine Lockdown**: Prohibido modificar `src/v7_engine`. Si falta una utilidad, implementarla localmente en la carpeta de la estrategia o en `03_RESEARCH_LAB/utils/`.
- **No Placeholders**: El código debe ser funcional para backtest inmediato.
