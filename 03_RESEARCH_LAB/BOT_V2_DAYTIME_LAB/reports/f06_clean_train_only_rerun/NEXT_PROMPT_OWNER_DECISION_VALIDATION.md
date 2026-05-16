# NEXT PROMPT: OWNER DECISION VALIDATION

**CONTEXTO:**
El owner ha llenado el documento `OWNER_DECISION_D1_D5_TEMPLATE.md` tomando las decisiones fundamentales D1-D5 sobre la definición de la estrategia F06, taxonomía de configuraciones, modelo de costos y resolución de métricas en el ledger. 

**OBJETIVO:**
1. Tomar el documento `OWNER_DECISION_D1_D5_TEMPLATE.md` completado y validar que todas las respuestas son consistentes, lógicas y seguras.
2. Verificar que la estrategia seleccionada existe en `STRATEGY_REGISTRY` y cumple con la firma de Phase 3.
3. Verificar que los parámetros de la estrategia son válidos según el código de la misma.
4. Validar la política de costos (spread, slippage, commission) y mapeo del ledger.
5. Preparar (PERO NO IMPLEMENTAR TODAVÍA) el esqueleto lógico para el Adapter de ejecución.

**REGLAS ABSOLUTAS:**
- NO implementar el adapter todavía.
- NO correr F06.
- NO correr backtests.
- NO tocar core/engine.
- NO tocar 2025/2026.
- NO hacer sweep de parámetros.

**TAREAS ESPERADAS DEL AGENTE:**
1. Leer y analizar minuciosamente el `OWNER_DECISION_D1_D5_TEMPLATE.md`.
2. Emitir un veredicto de validación: `OWNER_DECISION_VALIDATED` o `OWNER_DECISION_REJECTED`.
3. Si es aceptado, actualizar la documentación técnica del proyecto indicando que "F06" ha sido oficialmente definida en Phase 3, referenciando la estrategia y el hash de parámetros.
4. Mostrar un pseudo-código o diagrama del Adapter necesario que satisfaga la decisión D5 sin tocar el motor principal.
5. Indicar el siguiente paso seguro (Adapter Implementation).
