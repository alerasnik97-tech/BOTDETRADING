# P0/P1 Data & News Manifest Status
Fecha: 2026-05-14

## Estado de Inventario
- **Data Vault (`05_MARKET_DATA_VAULT`)**: EXISTE
- **News CSV (`news_eurusd_am_fortress_v3.csv`)**: EXISTE (Hash: 4F047DDA813D00E3882D3C6307060626A405C27B94F32BED29D99C432710BE67)
- **Manifest News (`NEWS_RESTORE_MANIFEST.json`)**: EXISTE
- **Coverage Audit**: EXISTE (v49_7_parallel_audit)
- **Hash Audit**: EXISTE (v49_7_parallel_audit)

## Gaps Identificados
- **DATA_MANIFEST local**: FALTA (El archivo existe en Git pero no hay una versión actualizada y detallada bar-by-bar en el vault local).
- **SCHEMA local**: EXISTE (Referencia en README).
- **Lineage Documentado**: PARCIAL (Se conoce el origen dukascopy/fortress pero falta el log de transformaciones V1-V7).

## Impacto
- **Bloquea Research**: NO (Datos verificados por hashes).
- **Bloquea Paper/Demo/Fondeo**: SÍ (Se requiere trazabilidad total y auditoría de latencia de datos antes de operar capital).

## Recomendación
Implementar un `DATA_MANIFEST.json` que incluya el `row_count` y `sha256` de cada mes de parquets para prevenir corrupciones silenciosas.
