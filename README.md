# BOT DE TRADING

Repositorio de research cuantitativo para forex intradia. La base actual y vigente del proyecto vive en [research_lab](C:/Users/alera/Desktop/BOT%20DE%20TRADING/research_lab).

## Estructura activa

- [research_lab](C:/Users/alera/Desktop/BOT%20DE%20TRADING/research_lab)
  Motor, estrategias, validacion, auditorias y runners actuales.
- [requirements.txt](C:/Users/alera/Desktop/BOT%20DE%20TRADING/requirements.txt)
  Dependencias minimas del laboratorio.
- [scripts/bootstrap_free_dukascopy.ps1](C:/Users/alera/Desktop/BOT%20DE%20TRADING/scripts/bootstrap_free_dukascopy.ps1)
  Bootstrap opcional para preparar data.
- `data_free_2020/prepared/EURUSD_M5.csv`
- `data_candidates_2022_2025/prepared/EURUSD_M5.csv`
  Dataset minimo versionado para corridas cloud.

## Estructura archivada

- [legacy](C:/Users/alera/Desktop/BOT%20DE%20TRADING/legacy)
  Scripts y prototipos anteriores preservados fuera del flujo actual.
- [docs/archive](C:/Users/alera/Desktop/BOT%20DE%20TRADING/docs/archive)
  Notas y documentacion historica.
- [docs/examples](C:/Users/alera/Desktop/BOT%20DE%20TRADING/docs/examples)
  Ejemplos auxiliares, como el CSV minimo de noticias.

## Quick start

Instalacion:

```bash
pip install -r requirements.txt
```

Backtest de una estrategia:

```bash
python -m research_lab.main run --strategy ema_trend_pullback
```

Corrida completa del laboratorio:

```bash
python -m research_lab.main run-all --pair EURUSD --start 2020-01-01 --end 2025-12-31 --data-dirs data_free_2020/prepared data_candidates_2022_2025/prepared --results-dir results/research_lab_robust --max-evals 8 --seed 42
```

Corrida ligera por tandas:

```bash
python -m research_lab.light_runner --strategies donchian_breakout_regime bollinger_mean_reversion_adx_low --pair EURUSD --start 2020-01-01 --end 2025-12-31 --data-dirs data_free_2020/prepared data_candidates_2022_2025/prepared --results-dir results/research_lab_light --phase1-evals 6 --phase2-evals 10 --seed 42
```

## Reglas operativas actuales

- instrumento principal: `EURUSD`
- timeframe principal: `M15`
- horario operativo: `11:00` a `19:00` America/New_York
- `main` es la rama estable
- nuevas tareas deben salir desde una rama nueva
- noticias actualmente: `OFF`

## Referencias utiles

- [research_lab/README.md](C:/Users/alera/Desktop/BOT%20DE%20TRADING/research_lab/README.md)
- [research_lab/BASELINE_LEVEL1.md](C:/Users/alera/Desktop/BOT%20DE%20TRADING/research_lab/BASELINE_LEVEL1.md)
- [docs/examples/news_example.csv](C:/Users/alera/Desktop/BOT%20DE%20TRADING/docs/examples/news_example.csv)
