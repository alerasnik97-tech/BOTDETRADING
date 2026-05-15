# P0/P1 Mock & Legacy Quarantine Plan
Fecha: 2026-05-14

## Análisis de Situación
Se ha detectado una superficie crítica de módulos que generan evidencia sintética para "testear" el pipeline de reportes. Aunque útiles para desarrollo, representan un riesgo de integridad si sus outputs se confunden con resultados reales de EURUSD.

## Plan de Cuarentena
### 1. Invalidación Inmediata
- Los reportes generados por `sweep_direct.py` y `walk_forward_runner.py` quedan marcados como **NON-CANONICAL**. No pueden ser usados para justificar inversiones o promociones de estrategias.
- Cualquier archivo CSV en `reports/` que contenga la palabra "MOCK" o "SYNTHETIC" queda fuera del alcance de la auditoría de Edge.

### 2. Aislamiento de Código (Post-V50B)
- Una vez finalizada la corrida V50B, los scripts de generación sintética en `src/v7_engine/` deben ser movidos a una subcarpeta `src/v7_engine/legacy_mocks/` o eliminados.
- Se debe prohibir la importación de `DummyNews` en cualquier runner destinado a producción o investigación seria.

### 3. Marcado de Evidencia
- Todo reporte oficial DEBE incluir el campo `data_source_integrity_hash` para verificar que proviene de parquets reales del Vault y no de generadores aleatorios.

## Conclusión
La existencia de generadores sintéticos en el CORE es una deuda técnica de alta gravedad. El laboratorio debe transicionar hacia una validación basada 100% en datos físicos e inmutables.
