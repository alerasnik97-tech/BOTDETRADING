# PHASE 48 — LESSONS LEARNED: EURUSD DAYTIME STRATEGY RESEARCH

## 1. Resumen Ejecutivo
Tras una auditoría profunda de 10 estrategias intradía para el par EURUSD durante el periodo 2020-2025, el laboratorio concluye que **no existe un edge estadístico neto** en las variantes de reversión a la media, rupturas de rango horario simple o continuación de tendencia basadas en indicadores clásicos bajo condiciones de costos institucionales.

## 2. Veredicto Final
**VEREDICTO: NO_DAYTIME_SECOND_STRATEGY_CANDIDATE_FOUND**

## 3. Principales Aprendizajes Cuantitativos

### A. El "Asesino de Edges": Fricción de Costos
- El spread institucional (1.2 pips) y el slippage conservador (0.2 pips) actúan como un filtro insuperable para estrategias de alta frecuencia o targets pequeños.
- **Regla de Oro**: Cualquier hipótesis con una Expectancy Bruta inferior a **0.15R** debe ser rechazada de inmediato, ya que los costos la llevarán a terreno negativo.

### B. Ineficiencia de la Sesión NY PM
- La estrategia `10_NY_PM_ROTATION` demostró que los rechazos en extremos de rango en la tarde de New York son trampas de liquidez. La falta de volumen institucional real facilita expansiones erráticas que rompen el mean reversion.

### C. Fallo de las Rupturas de Rango (ORB)
- `01_ORB_V1` y `03_NMCB_V1` confirmaron que el mercado de EURUSD es altamente eficiente en las aperturas. Las rupturas sin un contexto estructural superior (como los Fractal Sweeps de MANIPULANTE) tienen un winrate cercano al 50%, lo cual es insuficiente tras costos.

### D. Volatilidad Post-Noticia
- `09_POST_NEWS_DRIFT` estuvo cerca de la neutralidad (PF Neto 0.96), pero el riesgo de "whipsaws" (latigazos) y la ampliación del spread durante noticias de alto impacto la hacen operativamente peligrosa.

## 4. Por qué se rechazó cada estrategia (Resumen)
- **01-02 (ORB/LCF)**: Edge bruto marginal, winrate insuficiente.
- **03 (NMCB)**: Compresión sin dirección clara; PF Neto 0.76.
- **04-05 (AVE/MTLR)**: Sobreoperación; el mercado absorbe los niveles de soporte/resistencia estáticos.
- **06 (VWAP DMR)**: El VWAP como imán es real (PF Bruto 1.14), pero la distancia de reversión es menor que el costo de entrada.
- **07-08 (EMA/PDM)**: Falta de inercia; EURUSD tiende a rotar más que a seguir tendencias lineales en horario daytime.
- **09 (News Drift)**: Sensibilidad extrema al slippage; riesgo de cola no compensado.
- **10 (PM Rotation)**: Decaimiento absoluto; DD de -126R en escenarios de estrés.

## 5. Reglas para Futuras Fases
1. **Mandato Estructural**: No volver a testear estrategias basadas únicamente en ventanas horarias e indicadores (RSI, EMA, ADX).
2. **Filtro de Costos Previo**: Realizar una simulación de costos de 1.5 pips antes de proceder a la optimización profunda.
3. **Prioridad Fractal**: El aprendizaje confirma que el enfoque de MANIPULANTE (H1 Liquidity + M3 Confirmation) es superior por su capacidad de filtrar ruido intradía.

## 6. Siguiente Paso Recomendado
Cerrar el bloque Phase 48 y pivotar hacia la investigación de **Multi-Timeframe Structure (MTS)** que complemente la lógica actual de MANIPULANTE sin aumentar la frecuencia de forma artificial.
