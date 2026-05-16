# NEXT PROMPT: EURUSD Strategy Skeleton Implementation (Priority A Final - Wave 1)

Actúa como **Lead Quant Developer & FX Architecture Specialist**. Tu misión es implementar los esqueletos de señal para las estrategias de **Priority A Final** resultantes del arbitraje institucional post-Grok recovery.

## Objetivo
Codificar la lógica de señal y gestión de riesgo técnica para el lote inicial de estrategias en `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/strategies/`, asegurando compatibilidad con el motor v7.

## Estrategias a Implementar (Wave 1)
1. **MR-01 Anchor Elastic**: Reversión desde extremos del APM (Ancla de Precio Medio). Usar desviaciones de 1.8 - 2.2 SD.
2. **MR-02 VWAP Stretch Reversion**: Reversión desde ±2.25 SD del VWAP de apertura NY.
3. **TP-01 London-NY Momentum Pullback**: Pullback a EMA20 filtrado por:
   - Impulso previo de 5 barras > 1.5x ATR promedio.
   - ATR Percentile (últimas 200 barras) > 50.
4. **VE-ORB Volatility Expansion**: Ruptura del rango 07:00-08:00 NY con filtro ATR(14) > p65-70.

## Requisitos de Código
- **No Lookahead**: Todos los cálculos (VWAP, APM, ATR) deben basarse en `t-1` o ser acumulativos causales.
- **OHLCV-Only**: No incluir dependencias de noticias ni datos externos.
- **Fail-Closed**: Si faltan barras o el spread es anómalo, señal `0`.
- **Output Contract**: Exportar clase/función que devuelva `signal` {-1, 0, 1}, `sl_price` y `tp_price`.

## Instrucciones de Implementación
1. Crear rama: `research/EURUSD-wave1-skeletons-20260516`.
2. Para cada estrategia:
   - Crear archivo `.py` (ej: `strat_mr_01_anchor_elastic.py`).
   - Implementar indicadores anclados (VWAP NY 07:00).
   - Aplicar parámetros "frozen" indicados arriba.
3. **Unit Tests**: Crear smoke tests verificando que la señal es coherente con el OHLCV de entrada.

## Restricciones de Seguridad
- NO realizar backtests reales.
- NO tocar datos de 2025/2026 ni holdout.
- NO modificar `src/v7_engine`.
