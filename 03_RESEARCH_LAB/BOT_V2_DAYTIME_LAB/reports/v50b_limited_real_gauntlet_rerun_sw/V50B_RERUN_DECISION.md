# V50B RERUN FINAL DECISION

**Estado**: **RED_VALID_FAILURE**
**Fecha**: 2026-05-14

## Resumen de la Corrida
Se ha completado el Rerun seguro del Gauntlet V50B utilizando la arquitectura Ultra-Optimizada Single-Writer.

- **Integridad**: 100% (Lock verificado, Append-only verificado).
- **Mapeo de Ejecución**: Corregido (Long/Short verificado).
- **Datos Reales**: Noticias reales conectadas (504 eventos).
- **Blindaje TEST**: 0% Leakage (2025/2026 inaccesibles).

## Hallazgos
Ninguna de las 150 configuraciones evaluadas (F06, F08, F12) superó los umbrales de promoción institucional:
- **F06**: Muestra destellos de rentabilidad en Validación pero es inestable en Entrenamiento.
- **F08**: Altamente correlacionada con el ruido de la sesión NY; el Throttler bloqueó múltiples intentos.
- **F12**: Tras corregir el mapeo de dirección, el WR 100% desapareci revelando una estrategia con PF < 1.0.

## Veredicto
**BLOQUEADO**: El paso a V50C (Expansin) queda denegado para estas familias en su forma actual.
**RECOMENDACIÓN**: Pivotar hacia la reescritura de F01 (NY Window) y la optimización de parámetros de F06 utilizando el motor Ultra-Optimizado desarrollado en esta fase.

## Evidencia Física
- [V50B_RERUN_MASTER_RANKING.csv](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/results/V50B_RERUN_MASTER_RANKING.csv)
- [V50B_RERUN_TRADES.csv](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/trades/V50B_RERUN_TRADES.csv)
- [V50B_RERUN_ENGINE_CALL_PROOF.csv](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/engine_proof/V50B_RERUN_ENGINE_CALL_PROOF.csv)
