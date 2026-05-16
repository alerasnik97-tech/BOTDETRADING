# NEXT_PROMPT: F06_SAFE_ENGINE_ADAPTER_IMPLEMENTATION

## Contexto
El diseño del Safe Engine Adapter para el proceso de reconstrucción de evidencia F06 ha sido completado y auditado (`SAFE_ENGINE_ADAPTER_DESIGN_ONLY_REPORT.md`). Se ha superado el `SAFE_ENGINE_ADAPTER_IMPLEMENTATION_GATE` tras el cleanup de la polución local.

## Objetivo
Implementar el adaptador de motor seguro que permitirá ejecutar la reconstrucción de evidencia de F06/V50B bajo las reglas estrictas del contrato de salida institucional (Phase 3 Output Contract).

## Instrucciones para Antigravity
1. **Engine Discovery (Read-Only)**: Antes de escribir código, realizar el discovery final de los cargadores de datos de F06 y los parámetros exactos de `EngineConfig` requeridos.
2. **Adapter Implementation**: Crear el paquete `adapters/` en el pipeline F06 e implementar `phase3_f06_engine_adapter.py`.
3. **Governance Enforcement**: El adaptador debe:
   - Bloquear el acceso a 2025/2026.
   - Forzar el modo `TRAIN_ONLY`.
   - Implementar el guardado atómico (temp -> final).
   - Generar el manifiesto con hashes SHA256.
   - Validar el contrato de salida post-run.
4. **Unit Testing**: Crear la suite de tests para el adaptador.
5. **NO EJECUCIÓN REAL**: Esta fase es exclusivamente para la IMPLEMENTACIÓN y PRUEBAS UNITARIAS del adaptador. No se autoriza la corrida de F06 real sobre el dataset completo todavía.

## Prohibiciones
- NO modificar `research_lab/engine.py`.
- NO tocar Holdout (2025/2026).
- NO usar `git add .`.
