# NEXT PROMPT: EURUSD Strategy Skeleton Implementation (Priority A Final)

Actúa como **Lead Quant Developer & FX Architecture Specialist**. Tu misión es implementar los esqueletos de señal para las estrategias de **Priority A Final** resultantes del arbitraje institucional.

## Objetivo
Codificar la lógica de señal y gestión de riesgo técnica para el lote inicial de estrategias en `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/strategies/`, asegurando compatibilidad con el motor v7.

## Estrategias a Implementar (Wave 1)
1. **MR-01 Anchor Elastic**: Reversión desde extremos del APM (Ancla de Precio Medio).
2. **MR-02 VWAP Stretch Reversion**: Reversión desde ±2.25 SD del VWAP.
3. **VE-ORB Volatility Expansion**: Ruptura de rango de apertura (07:00-08:00 NY) con filtro ATR > p70.

## Requisitos de Código
- **No Lookahead**: Todos los cálculos deben basarse en `t-1`.
- **OHLCV-Only**: No incluir dependencias de noticias, sentimiento ni datos externos.
- **Fail-Closed**: Si el spread es anómalo o faltan barras, la señal debe ser `0` (Neutral).
- **Output Contract**: Cada archivo debe exportar una clase/función que devuelva:
  - `signal`: {-1, 0, 1}
  - `sl_price`: float
  - `tp_price`: float

## Instrucciones de Implementación
1. Crear una rama dedicada: `research/EURUSD-wave1-skeletons-20260516`.
2. Para cada estrategia:
   - Crear el archivo `.py` (ej: `strat_mr_01_anchor_elastic.py`).
   - Implementar los indicadores anclados (APM, VWAP, ATR Percentile).
   - Aplicar los parámetros congelados (round numbers) definidos en el backlog.
3. **Unit Tests**: Crear tests mínimos para verificar que la señal se dispara en condiciones extremas conocidas (Smoke tests).

## Restricciones de Seguridad
- NO realizar backtests reales (prohibida la ejecución masiva).
- NO tocar datos de 2025/2026 ni holdout.
- NO modificar `src/v7_engine`.
