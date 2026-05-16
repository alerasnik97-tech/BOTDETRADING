# NEXT_PROMPT: STRATEGY_RESEARCH_INTAKE_DEEP_READ

## Contexto
El laboratorio ha sido estabilizado y la documentación de investigación externa está catalogada en `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/`. El motor v7 es compatible con la interfaz `.signal()` y los tests de integridad pasan al 100%.

## Objetivo
Iniciar la extracción sistemática de hipótesis algorítmicas desde la documentación ingerida para poblar el backlog de investigación EURUSD Train-Only.

## Instrucciones para Antigravity
1. **Full Document Scan**: Leer detalladamente los 6 archivos ubicados en `original_files/`.
2. **Strategy Extraction**: Identificar estrategias concretas. Para cada una extraer:
   - Nombre propuesto.
   - Lógica de entrada (Indicadores, niveles, sesiones).
   - Lógica de salida (TP/SL, trailing, break-even).
   - Filtros (ATR, volumen, horario).
3. **Deduplication**: Si varias investigaciones apuntan a la misma lógica (ej. Sweep de Asia), consolidarlas en una única hipótesis superior.
4. **Hypothesis Backlog Generation**: Crear `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/hypothesis_backlog/EURUSD_HYPOTHESIS_BACKLOG.md` con la lista de estrategias candidatas.
5. **Normalization Plan**: Determinar qué estrategias son compatibles directamente con `engine.py` y cuáles requieren adaptadores nuevos.
6. **NO BACKTEST**: Prohibido ejecutar backtests en esta fase de lectura. Solo análisis teórico y diseño de hipótesis.

## Prohibiciones
- NO tocar Holdout (2025/2026).
- NO modificar el núcleo del motor.
- NO ejecutar código de las investigaciones sin previa sanitización.
