# External SCBI Research Harness

Harness externo de research para `SCBI_M5_GLOBAL`.

Objetivo:
- replicar afuera del core la logica real actual del runner productivo;
- correr backtests historicos masivos sin contaminar forward;
- comparar variantes controladas y auditables;
- escribir resultados nuevos solo en la carpeta externa del harness.

## Estructura

```text
external_scbi_research_harness/
  __init__.py
  config.py
  data_io.py
  matrix.py
  orchestrator.py
  reporting.py
  strategy.py
  outputs/
```

Scripts CLI en la raiz del workspace:
- `C:\Users\alera\Desktop\Bot\run_research_baseline.py`
- `C:\Users\alera\Desktop\Bot\run_research_matrix.py`
- `C:\Users\alera\Desktop\Bot\summarize_research.py`

## Que replica exactamente la baseline

`baseline_truth_model` replica la estrategia real actual del runner productivo:
- `EURUSD`
- sweep en `H1`
- niveles `PDH/PDL`, `Asia H/L`, `London H/L`
- long: `low < level` y `close > level`
- short: `high > level` y `close < level`
- confirmacion `M5` entre `+1h` y `+2h`
- primera confirmacion valida
- entrada long: `next_open + 0.3 pips`
- entrada short: `next_open`
- `SL = extremo sweep +/- 1 pip`
- `TP = 1.5R`
- `timeout = 4h`
- maximo `1` trade por dia
- filtro noticias simplificado sobre el `sweep_time`

No agrega `CHOCH`, `FVG`, `BE`, trailing ni filtros de volatilidad.

## Variantes externas permitidas

La matriz externa puede cambiar, sin tocar produccion:
- gestion: `TP`, `timeout`, buffer de `SL`, buffer de entrada long
- confirmacion: ventana, reclaim simple vs reclaim con body strength, primera vs mejor
- niveles: `PD`, `Asia`, `London`, todos
- noticias: sin filtro, `+/-30m`, `+/-60m`, cooldown post noticia

Todas esas variantes quedan etiquetadas como research externo.

## Inputs requeridos

Solo lectura sobre datasets canonicos existentes:
- `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_free_2020\prepared\EURUSD_H1.csv`
- `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_free_2020\prepared\EURUSD_M5.csv`
- `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_candidates_2022_2025\prepared\EURUSD_H1.csv`
- `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_candidates_2022_2025\prepared\EURUSD_M5.csv`
- `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data\news_eurusd_am_fortress_v3.csv`

## Outputs esperados

Baseline:
- `baseline_trades.csv`
- `baseline_sweep_audit.csv`
- `baseline_summary.json`
- `baseline_summary.md`

Matriz:
- `research_matrix_results.csv`
- `research_top_variants.csv`
- `research_baseline_vs_variants.md`
- `research_summary.json`

## Como ejecutarlo

Baseline exacta:

```powershell
python C:\Users\alera\Desktop\Bot\run_research_baseline.py
```

Matriz externa completa por ejes:

```powershell
python C:\Users\alera\Desktop\Bot\run_research_matrix.py --profile axis_scan
```

Smoke run de la matriz:

```powershell
python C:\Users\alera\Desktop\Bot\run_research_matrix.py --profile axis_scan --max-variants 5
```

Resumen sobre un CSV ya generado:

```powershell
python C:\Users\alera\Desktop\Bot\summarize_research.py --results-csv C:\Users\alera\Desktop\Bot\external_scbi_research_harness\outputs\matrix_axis_scan\research_matrix_results.csv
```

## Que hace cada script

`run_research_baseline.py`
- carga `H1`, `M5` y noticias canonicas;
- ejecuta `baseline_truth_model`;
- escribe el paquete base de auditoria.

`run_research_matrix.py`
- construye la matriz de variantes externas;
- corre cada variante sobre los mismos datasets;
- rankea por robustez multi-anual, drawdown, consistencia, sample size y expectancy;
- escribe CSV, JSON y Markdown comparativos.

`summarize_research.py`
- toma un `research_matrix_results.csv` existente;
- reordena y regenera reportes de resumen sin relanzar toda la matriz.

## Criterios de comparacion entre variantes

El ranking principal no prioriza profit bruto solamente. Usa:
- `year_positive_ratio`
- `max_drawdown`
- `yearly_total_r_std`
- `sample_size`
- `expectancy`

## Que NO toca

Este harness no modifica:
- `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
- feeder
- autopilot
- validator
- promotion
- chain
- bundle builder
- monitoring layer
- `operational_analytics.py`

Solo lee archivos canonicos existentes y escribe artefactos nuevos dentro de:
- `C:\Users\alera\Desktop\Bot\external_scbi_research_harness\outputs`
