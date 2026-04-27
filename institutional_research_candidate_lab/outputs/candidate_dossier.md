# Candidate Dossier

## Baseline exacta replicada

- Instrumento: `EURUSD`.
- H1 para sweep y niveles; M5 para confirmacion, entrada y salida.
- Niveles: `PDH/PDL`, `Asia H/L`, `London H/L`.
- Confirmacion baseline: `+1h a +2h`, primera vela M5 cuyo `close` queda del lado correcto del nivel.
- Entrada baseline: siguiente vela M5. Long `next_open + 0.3 pips`; short `next_open`.
- Riesgo minimo: `2.0 pips`.
- SL baseline: extremo del sweep `+-1 pip`.
- TP baseline: `1.5R`.
- Timeout baseline: `4 horas`.
- Maximo `1` trade por dia.
- Noticias baseline: filtro simplificado alrededor del sweep.

## Cobertura de research

- Variantes evaluadas: `21`.
- Baseline sample_size: `1606`.
- Baseline PF: `2.408`.
- Baseline expectancy: `0.4149R`.
- Baseline max_drawdown_R: `-8.3268R`.

## Top 10 candidatos

| rank | variant_id | ranking_score | PF | expectancy | max_drawdown_R | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.912715 | 2.4741 | 0.486 | -11.0 | ROBUST_RESEARCH_CANDIDATE |
| 2 | tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.912715 | 2.4741 | 0.486 | -11.0 | ROBUST_RESEARCH_CANDIDATE |
| 3 | tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_body_strength_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.870157 | 2.3984 | 0.396 | -9.2915 | ROBUST_RESEARCH_CANDIDATE |
| 4 | tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_london_only_news_sweep_plus_minus_30m | 0.870141 | 3.1813 | 0.4869 | -5.2823 | ROBUST_RESEARCH_CANDIDATE |
| 5 | tp_1p50_timeout_6h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.838706 | 2.3815 | 0.4478 | -8.2344 | ROBUST_RESEARCH_CANDIDATE |
| 6 | tp_2p00_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.823107 | 2.5579 | 0.4901 | -10.0682 | ROBUST_RESEARCH_CANDIDATE |
| 7 | tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_15m | 0.822425 | 2.3859 | 0.4131 | -9.5997 | RESEARCH_ONLY_MONITOR |
| 8 | tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_none | 0.815729 | 2.3846 | 0.4132 | -9.5997 | RESEARCH_ONLY_MONITOR |
| 9 | tp_1p50_timeout_2h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.805452 | 2.3031 | 0.3269 | -7.3401 | DO_NOT_PROMOTE |
| 10 | tp_1p00_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.80124 | 2.2485 | 0.323 | -8.0438 | DO_NOT_PROMOTE |

## Top 3 candidatos robustos

### 1. tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m
- Por que entra al top: `ranking_score=0.912715`, `year_positive_ratio=1.0`, `sample_size=1638`.
- Mejora vs baseline: `PF +0.0661`, `expectancy +0.0711R`.
- Empeora vs baseline: `drawdown absoluto +2.6732R`, `timeout_exit_rate -0.1515`.
- Veredicto: `ROBUST_RESEARCH_CANDIDATE`.
- Siguiente paso: Mantener como RESEARCH_ONLY y dejarlo listo para futura shadow line.

### 2. tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m
- Por que entra al top: `ranking_score=0.912715`, `year_positive_ratio=1.0`, `sample_size=1638`.
- Mejora vs baseline: `PF +0.0661`, `expectancy +0.0711R`.
- Empeora vs baseline: `drawdown absoluto +2.6732R`, `timeout_exit_rate -0.1515`.
- Veredicto: `ROBUST_RESEARCH_CANDIDATE`.
- Siguiente paso: Mantener como RESEARCH_ONLY y dejarlo listo para futura shadow line.

### 3. tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_body_strength_pick_first_levels_all_levels_news_sweep_plus_minus_30m
- Por que entra al top: `ranking_score=0.870157`, `year_positive_ratio=1.0`, `sample_size=1588`.
- Mejora vs baseline: `PF -0.0096`, `expectancy -0.0189R`.
- Empeora vs baseline: `drawdown absoluto +0.9647R`, `timeout_exit_rate +0.043`.
- Veredicto: `ROBUST_RESEARCH_CANDIDATE`.
- Siguiente paso: Mantener como RESEARCH_ONLY y dejarlo listo para futura shadow line.

## Candidatos que NO deben promoverse

- `tp_1p50_timeout_2h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`: `PF=2.3031`, `expectancy=0.3269`, `sample_size=1606`, veredicto `DO_NOT_PROMOTE`.
- `tp_1p00_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`: `PF=2.2485`, `expectancy=0.323`, `sample_size=1606`, veredicto `DO_NOT_PROMOTE`.
- `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_asia_only_news_sweep_plus_minus_30m`: `PF=1.7975`, `expectancy=0.2342`, `sample_size=1306`, veredicto `DO_NOT_PROMOTE`.
- `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_best_levels_all_levels_news_sweep_plus_minus_30m`: `PF=1.0789`, `expectancy=0.0296`, `sample_size=1606`, veredicto `DO_NOT_PROMOTE`.
- `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_pd_only_news_sweep_plus_minus_30m`: `PF=0.9909`, `expectancy=-0.004`, `sample_size=1108`, veredicto `DO_NOT_PROMOTE`.

## Mejor candidato para una futura shadow line

- Candidato propuesto: `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`.
- Veredicto: `ROBUST_RESEARCH_CANDIDATE`.
- Razones principales: `PF=2.4741`, `expectancy=0.486R`, `max_drawdown_R=-11.0R`, `sample_size=1638`.

## Siguiente paso recomendado

- Mantener todo en `RESEARCH_ONLY / NO_PRODUCTION` y, si se aprueba institucionalmente, usar solo el mejor candidato como futura shadow line externa.
