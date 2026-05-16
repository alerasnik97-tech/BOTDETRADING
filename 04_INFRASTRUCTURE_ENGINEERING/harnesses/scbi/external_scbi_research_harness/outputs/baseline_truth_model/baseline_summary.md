# Baseline Truth Model

## Lo mas importante

- Replica externa del runner real actual sin tocar produccion.
- Variante: `baseline_truth_model`.
- Periodo: `2020-01-01` -> `2025-12-31`.

## Parametros replicados

- TP fijo: `1.5R`.
- Timeout: `4h`.
- Buffer SL: `1.0 pips`.
- Buffer entrada long: `0.3 pips`.
- Ventana confirmacion: `+1h_+2h`.
- Confirmacion: `Close M5 del lado correcto del nivel`.
- Seleccion de confirmacion: `Primera confirmacion elegible`.
- Niveles habilitados: `all_levels`.
- Filtro noticias: `Bloqueo por noticia +/-30m alrededor del sweep`.

## Metricas

- sample_size: `1606`
- win_rate: `0.6202`
- PF: `2.408`
- expectancy: `0.4149R`
- avg_hold_minutes: `139.99`
- timeout_exit_pct: `0.3487`
- max_drawdown: `-8.3268R`

## Conteos de auditoria

- sweeps_considered: `12176`
- trades_executed: `1606`
- news_blocked: `824`
- daily_limit_skipped: `9546`
- no_scbi_found: `172`
- invalid_risk: `13`
