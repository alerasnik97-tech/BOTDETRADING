# Baseline vs Variants

- perfil_matriz: `existing_results`
- variantes_evaluadas: `21`

## Baseline

- variant_id: `baseline_truth_model`
- sample_size: `1606`
- PF: `2.408`
- expectancy: `0.4149R`
- max_drawdown: `-8.3268R`
- year_positive_ratio: `1.0`

## Top Variants

| variant_id | ranking_score | sample_size | pf | expectancy | max_drawdown | year_positive_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.937803 | 1638 | 2.4741 | 0.486 | -11.0 | 1.0 |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.937803 | 1638 | 2.4741 | 0.486 | -11.0 | 1.0 |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_body_strength_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.896288 | 1588 | 2.3984 | 0.396 | -9.2915 | 1.0 |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_london_only_news_sweep_plus_minus_30m | 0.862933 | 1259 | 3.1813 | 0.4869 | -5.2823 | 1.0 |
| tp_1p50_timeout_6h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.84646 | 1606 | 2.3815 | 0.4478 | -8.2344 | 1.0 |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_none | 0.820848 | 1615 | 2.3846 | 0.4132 | -9.5997 | 1.0 |
| tp_1p50_timeout_2h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.817451 | 1606 | 2.3031 | 0.3269 | -7.3401 | 1.0 |
| tp_1p00_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.816873 | 1606 | 2.2485 | 0.323 | -8.0438 | 1.0 |
| tp_2p00_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.814081 | 1606 | 2.5579 | 0.4901 | -10.0682 | 1.0 |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_post_news_cooldown_60m | 0.813603 | 1609 | 2.4001 | 0.4171 | -9.5997 | 1.0 |
| baseline_truth_model | 0.794465 | 1606 | 2.408 | 0.4149 | -8.3268 | 1.0 |
| tp_1p50_timeout_4h_sl_0p5_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.794234 | 1605 | 2.3906 | 0.4159 | -8.0859 | 1.0 |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_60m | 0.793186 | 1602 | 2.3963 | 0.4126 | -8.3268 | 1.0 |
| tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_asia_only_news_sweep_plus_minus_30m | 0.79272 | 1306 | 1.7975 | 0.2342 | -11.6175 | 1.0 |
| tp_1p25_timeout_4h_sl_1p0_longbuf_0p3_win_1_2_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m | 0.784275 | 1606 | 2.3042 | 0.3666 | -8.131 | 1.0 |

## Criterios de ranking

- `year_positive_ratio`: mayor es mejor.
- `max_drawdown`: menor drawdown absoluto es mejor.
- `yearly_total_r_std`: menor dispersion anual es mejor.
- `sample_size`: mayor es mejor.
- `expectancy`: mayor es mejor.

No se rankea por profit bruto solamente.
