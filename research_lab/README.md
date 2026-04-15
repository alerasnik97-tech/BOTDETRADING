# Research Lab Robust

Laboratorio reproducible para evaluar familias de estrategias intradia 100% programables sobre `EURUSD` usando `M15` como timeframe principal.

## Estrategias activas

- `ema_trend_pullback`
- `bollinger_mean_reversion_adx_low`
- `donchian_breakout_regime`
- `keltner_squeeze_breakout`
- `supertrend_ema_filter`

## Principios

- prioriza robustez antes que retorno bruto
- backtest completo `2020-01-01 -> 2025-12-31`
- walk-forward rolling default `24m IS + 6m OOS`
- walk-forward alternativo `36m IS + 6m OOS`
- costos realistas:
  - spread configurable
  - slippage adverso
  - comision roundturn por lote
- penalizacion fuerte por muestra insuficiente

## Instalacion

```bash
pip install -r requirements.txt
```

## Baseline oficial

- Baseline Nivel 1 congelada en [BASELINE_LEVEL1.md](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/research_lab/BASELINE_LEVEL1.md)
- `normal_mode` conserva la baseline aprobada
- `conservative_mode` agrega una capa opcional de realismo conservador auditable

## Runtime local validado

En este proyecto el rerun de auditoria/tests usa un runtime local dentro del workspace:

```powershell
$env:PYTHONPATH='C:\Users\alera\Desktop\BOT DE TRADING\.pkg'
$env:MPLCONFIGDIR='C:\Users\alera\Desktop\BOT DE TRADING\.mplconfig'
$env:TEMP='C:\Users\alera\Desktop\BOT DE TRADING\.tmp'
$env:TMP='C:\Users\alera\Desktop\BOT DE TRADING\.tmp'
```

Si esas variables no estan definidas, el entorno puede volver a fallar por App Control sobre DLLs del runtime global.

## Reglas duras del loader

- Los CSV preparados deben traer offset timezone explicito en el indice.
- El loader rechaza cualquier CSV timezone-naive.
- El formato aceptado es compatible con timestamps como `2022-01-03 11:00:00-05:00`.

## Noticias

- Fuente raw auditada: `data/forex_factory_cache.csv`
- Dataset limpio derivado: `data/news_eurusd_m15_validated.csv`
- Dataset de auditoria: `data/news_eurusd_m15_audit.csv`
- Estado operativo actual: `source_approved=true` para el dataset validado derivado
- La fuente raw original sigue rechazada por timestamps defectuosos
- Si se quiere subir de nivel, el siguiente candidato serio es Trading Economics con export/API en UTC y validacion contra horarios oficiales

## Modos de ejecucion

- La fuente historica base es `OHLC BID`.
- El motor modela ASK, spread, slippage y comision de forma sintetica y auditable.
- `normal_mode` usa `cost_profile=base` e `intrabar_policy=standard`.
- `conservative_mode` usa `cost_profile=stress` e `intrabar_policy=conservative`.
- Si `SL` y `TP` caen dentro de la misma vela, `standard` respeta `intrabar_exit_priority` y `conservative` fuerza peor caso con slippage extra.
- Esto sirve para research serio y comparacion consistente, pero no reemplaza un simulador con `ASK` historico real o tick data.

## Correr una estrategia

```bash
python -m research_lab.main run --strategy ema_trend_pullback
```

Modo conservador:

```bash
python -m research_lab.main run --strategy ema_trend_pullback --execution-mode conservative_mode
```

## Optimizar / evaluar una estrategia

```bash
python -m research_lab.main optimize --strategy donchian_breakout_regime --max-evals 8 --seed 42
```

## Correr todo el laboratorio

```bash
python -m research_lab.main run-all --pair EURUSD --start 2020-01-01 --end 2025-12-31 --data-dirs data_free_2020/prepared data_candidates_2022_2025/prepared --results-dir results/research_lab_robust --max-evals 8 --seed 42
```

## Corrida liviana por tandas

```bash
python -m research_lab.light_runner --strategies donchian_breakout_regime bollinger_mean_reversion_adx_low --pair EURUSD --start 2020-01-01 --end 2025-12-31 --data-dirs data_free_2020/prepared data_candidates_2022_2025/prepared --results-dir results/research_lab_light --phase1-evals 6 --phase2-evals 10 --seed 42
```

## Salidas

Por estrategia:

- `summary.json`
- `trades.csv`
- `monthly_stats.csv`
- `yearly_stats.csv`
- `optimization_results.csv`
- `equity_curve.csv`
- `walkforward_default.csv`
- `walkforward_alt.csv`
- graficos `.png`

En la raiz del resultado:

- `strategy_ranking.csv`
- `comparative_table.csv`
- `top3_finalistas.md`
- `autopsia_perdedores.md`
- `recomendacion_final.md`
- graficos comparativos `.png`

## Bundle para ChatGPT

El ultimo resultado completo queda siempre en:

- [000_PARA_CHATGPT.zip](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/000_PARA_CHATGPT.zip)

## Adquisicion de mejores fuentes

Precios:

- auditoria local de fuentes: `python -m research_lab.price_source_audit --pair EURUSD`
- recomendacion principal: Dukascopy con `tick bid+ask` o `M1 bid+ask`
- validacion secundaria recomendada: TrueFX para contrastar spreads y timestamps
- descarga automatica de `M1 bid+ask`: `python -m research_lab.dukascopy_m1_download --pair EURUSD --start 2024-10-01 --end 2025-03-31 --output-dir data_precision_raw/dukascopy`
- integracion de fuente real de alta precision: `python -m research_lab.high_precision_import --source-type dukascopy_m1_bid_ask --print-schema`

Noticias:

- importador Trading Economics: `python -m research_lab.news_tradingeconomics --input data/tradingeconomics_calendar.json --output data/news_te_validated.csv --audit data/news_te_audit.csv --summary data/news_te_summary.json`
- el importador asume el campo `Date` en UTC, alineado con la documentacion oficial de Trading Economics
- el dataset generado queda en el formato canonico que ya entiende `research_lab.news_filter`
- guia operativa completa: [docs/SOURCE_INTEGRATION.md](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/docs/SOURCE_INTEGRATION.md)
