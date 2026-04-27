# Baseline vs Variants

## Baseline exacta replicada

- variant_id: `baseline_truth_model`
- sample_size: `1606`
- PF: `2.408`
- expectancy: `0.4149R`
- max_drawdown_R: `-8.3268R`

## Top 10 variantes

| variant_id | ranking_score | sample_size | PF | expectancy | max_drawdown_R | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.912715 | 1638 | 2.4741 | 0.486 | -11.0 | ROBUST_RESEARCH_CANDIDATE |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.912715 | 1638 | 2.4741 | 0.486 | -11.0 | ROBUST_RESEARCH_CANDIDATE |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_body_strength_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.870157 | 1588 | 2.3984 | 0.396 | -9.2915 | ROBUST_RESEARCH_CANDIDATE |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_london_only_news_sweep_plus_minus_30m | 0.870141 | 1259 | 3.1813 | 0.4869 | -5.2823 | ROBUST_RESEARCH_CANDIDATE |
| tp_1p50_timeout_6h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.838706 | 1606 | 2.3815 | 0.4478 | -8.2344 | ROBUST_RESEARCH_CANDIDATE |
| tp_2p00_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.823107 | 1606 | 2.5579 | 0.4901 | -10.0682 | ROBUST_RESEARCH_CANDIDATE |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_15m | 0.822425 | 1614 | 2.3859 | 0.4131 | -9.5997 | RESEARCH_ONLY_MONITOR |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_none | 0.815729 | 1615 | 2.3846 | 0.4132 | -9.5997 | RESEARCH_ONLY_MONITOR |
| tp_1p50_timeout_2h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.805452 | 1606 | 2.3031 | 0.3269 | -7.3401 | DO_NOT_PROMOTE |
| tp_1p00_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.80124 | 1606 | 2.2485 | 0.323 | -8.0438 | DO_NOT_PROMOTE |

## Criterio de ranking

- `year_positive_ratio`
- `max_drawdown_R`
- `yearly_total_R_std`
- `sample_size`
- `expectancy`
- `PF`
