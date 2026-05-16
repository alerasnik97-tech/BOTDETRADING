# Shadow Candidate Spec

- Estado: `RESEARCH_ONLY` / `NO_PRODUCTION`.
- variant_id: `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`.
- levels: `all_levels` (PD + Asia + London).
- confirmation_window: `+0h_+1h`.
- confirmation_pick: `first`.
- confirmation_mode: `close_reclaim`.
- long_entry_buffer_pips: `0.3`.
- short_entry_buffer_pips: `0.0`.
- sl_buffer_pips: `1.0`.
- tp_r: `1.5`.
- timeout_hours: `4`.
- news_mode: `sweep_plus_minus_30m`.

## Reglas exactas

- Sweep long: `low < nivel` y `close > nivel`.
- Sweep short: `high > nivel` y `close < nivel`.
- Confirmacion M5: dentro de la ventana indicada, segun el modo y pick indicados.
- Entrada: apertura de la vela M5 siguiente a la confirmacion.
- Riesgo minimo: `2.0 pips`.
- SL: extremo del sweep mas el buffer configurado.
- TP: multiple fijo `tp_r` sobre el riesgo.
- Timeout: cierre por tiempo a `timeout_hours`.
- Limite diario: `1 trade por dia`.
- Noticias: filtro simplificado sobre `sweep_time` segun `news_mode`.

No integrar a produccion sin aprobacion institucional posterior.
