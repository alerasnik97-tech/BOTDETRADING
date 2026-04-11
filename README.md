# BOT DE TRADING

Flujo principal actual: modelo híbrido simple por régimen para `EURUSD`.

## Arquitectura activa

- [simple_session_bot.py](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/simple_session_bot.py)
- [config.py](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/config.py)
- [news_filter.py](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/news_filter.py)
- [report.py](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/report.py)

## Lógica actual

- base `M5`
- contexto `H1` resampleado desde `M5`
- ventana fija:
  - entradas `11:00` a `17:30` NY
  - cierre forzado total `19:00` NY
- clasificador H1:
  - `EMA50`
  - `EMA200`
  - `ADX14`
  - progreso de tendencia reciente en múltiplos de `ATR_H1`
- un solo módulo activo por vez:
  - `breakout` en `trend_up` / `trend_down`
  - `range_mr` en `range_weak`
- `news guard`, `shock guard`, `spread guard`
- `daily circuit breaker`

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias activas:

- `python`
- `pandas`
- `numpy`

## Datos

Por defecto usa:

- `data_free_2020/prepared`
- `data_candidates_2022_2025/prepared`

Con eso cubre `EURUSD` entre `2020-01-01` y `2025-12-31`.

## Backtest

```bash
python simple_session_bot.py run --pair EURUSD --start 2020-01-01 --end 2025-12-31 --data-dirs data_free_2020/prepared data_candidates_2022_2025/prepared
```

Ejemplo sin noticias:

```bash
python simple_session_bot.py run --disable-news
```

## Optimización compacta

```bash
python simple_session_bot.py optimize --pair EURUSD --start 2020-01-01 --end 2025-12-31 --data-dirs data_free_2020/prepared data_candidates_2022_2025/prepared --max-combinations 18
```

La grilla compacta prueba solo packs representativos sobre:

- `model_mode`:
  - `hybrid`
  - `range_only`
  - `breakout_only`
- `adx_trend_min`
- `ema200_slope_lookback`
- `trend_progress_atr_min`
- `range_rsi_period`
- `bb_std`
- `range_rsi_low/high`
- `range_stop_atr`
- `range_be_enabled`
- `breakout_stop_mode`
- `breakout_stop_atr`
- `breakout_target_rr`
- `breakout_be_enabled`
- `breakout_enabled`
- `cooldown_bars`
- `daily_loss_limit_r`

## Parámetros manuales útiles

```bash
python simple_session_bot.py run --model-mode range_only --adx-trend-min 18 --ema200-slope-lookback 5 --trend-progress-atr-min 0.8 --range-rsi-period 9 --bb-std 2.0 --range-rsi-low 35 --range-rsi-high 65 --range-stop-atr 1.0 --range-target-rr 1.2 --range-be-enabled --cooldown-bars 3 --daily-loss-limit-r 1.5
```

```bash
python simple_session_bot.py run --model-mode breakout_only --compression-bars 6 --compression-atr-mult 1.0 --breakout-candle-atr-max 1.2 --breakout-stop-mode compression_stop --breakout-stop-atr 1.0 --breakout-target-rr 1.8 --breakout-be-enabled
```

## CSV de noticias

Por defecto usa:

- `data/forex_factory_cache.csv`

Columnas mínimas:

- `DateTime`
- `Currency`
- `Impact`
- `Event`

Ejemplo:

- [news_example.csv](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/news_example.csv)

## Salidas para ChatGPT

La carpeta visible para subir a ChatGPT es:

- [000_PARA_CHATGPT](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/000_PARA_CHATGPT)

Aviso directo:

- [ABRIR_000_PARA_CHATGPT.txt](/C:/Users/alera/Desktop/BOT%20DE%20TRADING/ABRIR_000_PARA_CHATGPT.txt)

Cada corrida exporta:

- `trades.csv`
- `monthly_stats.csv`
- `yearly_stats.csv`
- `hourly_stats.csv`
- `weekday_stats.csv`
- `summary.json`
- `equity_curve.csv`
- `optimization_results.csv`

## Estado actual

La v2 híbrida refinó la arquitectura y dejó el proyecto simple de iterar, pero no mejoró el edge frente al híbrido anterior. La evidencia actual favorece seguir usando este framework como base de research, no como sistema listo para producción.
