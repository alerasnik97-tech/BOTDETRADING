# Integracion de Fuentes Reales

Este proyecto mantiene la base actual intacta y agrega rutas opcionales para absorber mejores insumos cuando existan.

## 1. Precios Dukascopy de alta precision

### Opcion recomendada

- mejor opcion: `tick bid+ask`
- segunda opcion: `M1 bid+ask`

### Carpeta recomendada

- crudo: `data_precision_raw/dukascopy/`
- canonico generado: `data_precision/dukascopy/`

### Formato esperado si traes M1 bid+ask

Archivos exactos:

- `EURUSD_M1_BID.csv`
- `EURUSD_M1_ASK.csv`

Esquema esperado:

- primer campo = timestamp con timezone explicito
- columnas requeridas:
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`

Ejemplo de timestamp aceptado:

- `2025-01-02 11:00:00-05:00`

### Formato esperado si traes tick bid+ask

Archivos exactos:

- `EURUSD_TICK_BID.csv`
- `EURUSD_TICK_ASK.csv`

Esquema esperado:

- primer campo = timestamp con timezone explicito
- columnas requeridas:
  - `price`
  - `volume`

### Comandos

Ver esquema esperado:

```powershell
python -m research_lab.high_precision_import --source-type dukascopy_m1_bid_ask --print-schema
python -m research_lab.high_precision_import --source-type dukascopy_tick_bid_ask --print-schema
```

Descargar automaticamente Dukascopy M1 bid+ask:

```powershell
python -m research_lab.dukascopy_m1_download --pair EURUSD --start 2024-10-01 --end 2025-03-31 --output-dir data_precision_raw/dukascopy
```

Integrar M1 bid+ask:

```powershell
python -m research_lab.high_precision_import --pair EURUSD --source-type dukascopy_m1_bid_ask --input-dir data_precision_raw/dukascopy --output-dir data_precision/dukascopy
```

Integrar tick bid+ask:

```powershell
python -m research_lab.high_precision_import --pair EURUSD --source-type dukascopy_tick_bid_ask --input-dir data_precision_raw/dukascopy --output-dir data_precision/dukascopy
```

### Resultado esperado

Se generan:

- `EURUSD_M1_BID.csv`
- `EURUSD_M1_ASK.csv`
- `EURUSD_M1_MID.csv`
- `EURUSD_high_precision_manifest.json`

Notas:

- `MID` se deriva como promedio de `BID` y `ASK` reales
- esta integracion no cambia el motor automaticamente
- sirve para dejar lista la materia prima para una siguiente fase de precision

## 2. Trading Economics

### Carpeta recomendada

- export crudo: `data/news_imports/`

### Formatos de entrada soportados

- `JSON`
- `CSV`

### Campos esperados de Trading Economics

- `CalendarId`
- `Date`
- `Country`
- `Category`
- `Event`
- `Importance`
- `Currency`
- `Source`
- `SourceURL`

Supuesto operativo:

- `Date` viene en `UTC`

### Comando de importacion

```powershell
python -m research_lab.news_tradingeconomics --input data/news_imports/tradingeconomics_calendar.json --output data/news_te_validated.csv --audit data/news_te_audit.csv --summary data/news_te_summary.json
```

### Salida canonica esperada

- `event_id`
- `event_name_normalized`
- `currency`
- `impact_level`
- `timestamp_original`
- `timezone_original`
- `timestamp_utc`
- `timestamp_ny`
- `source_name`
- `dedupe_key`
- `validation_status`

### Regla para encender noticias

Noticias solo deben pasar a `ON` si el dataset limpio:

- pasa la validacion dura de horarios
- no presenta desplazamientos sistematicos
- cubre correctamente familias clave:
  - NFP
  - Unemployment Rate
  - CPI
  - Core CPI
  - Retail Sales
  - Core Retail Sales
  - GDP
  - PPI
  - ISM Manufacturing
  - ISM Services
  - FOMC
  - ECB relevantes

Si eso no se cumple, `news` debe seguir en `OFF`.

## 3. Orden recomendado

1. conseguir `Dukascopy M1 bid+ask` o `tick bid+ask`
2. integrarlo y validar manifest
3. conseguir export real de `Trading Economics`
4. importarlo al formato canonico
5. recien despues decidir si `news ON` es defendible
