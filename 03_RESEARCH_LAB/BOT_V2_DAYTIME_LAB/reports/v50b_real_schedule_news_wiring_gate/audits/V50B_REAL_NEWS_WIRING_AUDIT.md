# V50B REAL NEWS WIRING AUDIT

**Objetivo**: Certificar la conexión con el calendario económico real.

## Hallazgos de Datos
- **Archivo**: `news_eurusd_am_fortress_v3.csv`
- **Rango Detectado**: Inicia en 2020-01. Cubre los periodos de TRAIN (2022) y VAL (2023, 2024).
- **Columnas Críticas**:
  - `timestamp_utc`: Formato ISO 8601 con offset.
  - `currency`: EUR, USD.
  - `impact_level`: HIGH.
  - `event_name_normalized`: Identificador del evento.

## Plan de Cableado (Sin tocar Core)
1. Cargar el CSV usando `pandas`.
2. Convertir `timestamp_utc` a `datetime` (UTC naive para compatibilidad con el motor).
3. Instanciar `NewsCalendar` de `src.v7_engine.news_filter`.
4. Poblar con objetos `NewsEvent`.
5. Pasar esta instancia al `UnifiedV7Engine`.

## Verificación de Cobertura
- El archivo contiene eventos de alto impacto para EUR y USD.
- Se ha validado que el motor utiliza `is_blocked_by_news` correctamente sobre estos eventos.

**Veredicto**: REAL_NEWS_WIRED_READY.
