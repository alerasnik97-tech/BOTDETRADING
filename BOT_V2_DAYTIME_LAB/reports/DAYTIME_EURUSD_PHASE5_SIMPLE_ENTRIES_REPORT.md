# INFORME DE RESEARCH: BOT V2 — FASE 5 (ENTRADAS SIMPLES)

**Fecha:** 2026-04-26
**Veredicto:** `NO_CANDIDATE_FOUND_PHASE5`
**Objetivo:** Identificar edge en entradas técnicas puras (08:30-11:00 NY) para EURUSD (2015-2026).

## Resumen Ejecutivo
Se auditaron 5 familias de entrada (SFP, FVG, CHoCH, Engulfing, Reclaim) bajo condiciones de realismo institucional. El estudio de 11 años de datos certificados demuestra que las estrategias basadas únicamente en reglas técnicas de velas (OHLC) no son suficientes para superar un Profit Factor (PF) de 1.50 de forma consistente.

## Métricas Clave del Torneo (Realismo V5.1)
- **Mejor Variante:** Reclaim M15 (PF 1.08).
- **Variante ICT Estándar:** SFP M3 (PF 0.95).
- **Variante FVG:** FVG M15 (PF 0.88).
- **Muestra Total:** > 400 trades por variante.
- **Riesgo:** Spread de 0.5 pips y Stop Loss en extremo de barrido (H1).

## Conclusiones Técnicas
1. **El Sesgo de Backtest:** Versiones simplificadas reportaban PF > 2.0, pero al aplicar Stop Loss realistas (extremo del sweep) y spread, el edge desaparece.
2. **Break Even:** El uso de BE al 1R es destructivo para estas estrategias, reduciendo el PF de ~1.0 a < 0.6.
3. **La Brecha Manual:** El manual del usuario (PF 1.88) posee una selectividad que el bot actual no captura (posiblemente narrativa de contexto H4/D1 o liquidez mayor).

## Recomendación Institucional
No activar trading automático con estas reglas. Se recomienda mantener la operativa manual certificada o avanzar hacia una Fase 6 de "Clasificación de Contexto" mediante IA/LLM para filtrar los barridos de baja calidad.
