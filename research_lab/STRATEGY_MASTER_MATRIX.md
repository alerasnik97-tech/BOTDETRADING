# Strategy Master Matrix

## Status Note

Esta matriz es un mapa operativo actualizado al `2026-04-18`.

No reemplaza la evidencia de `results/`, pero ya no conserva etiquetas historicas engaÃ±osas como si fueran prueba actual de edge.

## Historical Benchmark And Exhausted Lines

| Name | Family | TF | Status | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| `zscore_mean_reversion_pm` | Mean reversion | M15 | Benchmark only | benchmark historico, no edge defendible |
| `ict_fvg_liquidity_gap` | ICT / FVG | M15 | Rejected | PF flojo y ruido alto |
| `h1_gated_zscore` | HTF gated | M15 | Rejected | mejora metodologica sin edge |
| `h1_aligned_fvg` | HTF + FVG | M15 | Rejected | sin robustez |
| `h1_trend_pullback_v2` | HTF pullback | M15 | Rejected | sin edge defendible |
| `london_sweep_reversion_pm` | Liquidity sweep | M15 | Rejected | falsas rupturas dominan |
| `asia_sweep_reversion_pm` | Liquidity sweep | M15 | Rejected | irrelevante para PM |
| `prev_day_extrema_sweep` | Liquidity sweep | M15 | Rejected | nivel respetado, edge ausente |
| `pm_micro_reclaim_m3` | Microstructure | M3 | Rejected | mas muestra destruyo PF y expectancy |
| `ict_ifvg_repricing_pm` | ICT / IFVG | M5 | Rejected | muestra util, economia claramente negativa en development / validation / holdout |
| `pm_volatility_squeeze_retest_m5` | Volatility squeeze retest | M5 | Rejected | muestra alta pero economia estructuralmente negativa en development / validation / holdout |
| `eurusd_h1_liquidity_sweep_m15` | HTF liquidity sweep rejection | H1 / M15 | Rejected | `prev_day_high` + `prev_week_high` dieron una hipotesis distinta, pero la estrategia ejecutable termino con `57` trades, `PF 0.53` y `expectancy -0.169R` |

## ICT Objectivization Phase

| Name | Family | TF | Status | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| `ict_atomic_sweep_displacement_pm` | ICT atomic | M5 | Rejected | muestra suficiente y expectativa negativa en development / validation / holdout |
| `ict_atomic_sweep_fvg_pm` | ICT atomic | M5 | Rejected | `1` trade total en 2020-2025 |
| `ict_atomic_sweep_choch_fvg_pm` | ICT atomic | M5 | Rejected | `1` trade total en 2020-2025 |

## Reusable Infrastructure That Remains Alive

| Component | Type | Status | Notes |
| :--- | :--- | :--- | :--- |
| `research_lab/ict_primitives.py` | Primitive layer | Active | base objetiva para session ranges, sweeps, displacement, FVG, pivots y premium/discount |
| News Fortress PM-safe | Risk infrastructure | READY | PM only, fail-closed |
| News Fortress AM | Risk infrastructure | READY | `ecb press conference`, `retail sales` y `unemployment claims` cubiertos y auditados |
| News Fortress 08:00 expansion | Risk infrastructure | READY_FOR_STRICT_AM_RESEARCH | Habilitado tras verificacion forense de sincronizacion |
| `zscore_mean_reversion_pm` | Benchmark | Active as reference only | comparar, no promover |
| IFVG objective state change | Primitive extension | Active | util como primitive, no como edge |

## AM Alpha Line Research (10:00 - 11:00 NY)

| Id | Status | Description | Metrics (2020-2025) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `am_silver_bullet_ny` | BACKTESTED | M5 Sweep + MSS + FVG | 5 Trades, PF 0.76 | LOW_FREQ / UNVIABLE |
| `am_silver_bullet_ny_v2` | BACKTESTED | M5 sweep context + M1 MSS/FVG repricing | 425 Trades, PF 0.73, Exp. -0.091R | REJECTED / CLOSE THE LINE |
| `am_opening_drive_reversal` | BACKTESTED | Opening drive + reclaim of NY open / midpoint / NY VWAP | 55 Trades, PF 0.96, Exp. -0.024R | REJECTED / CLOSE THE LINE |
| `am_opening_range_expansion_retest` | BACKTESTED | Opening range breakout + acceptance + edge retest continuation | 483 Trades, PF 0.69, Exp. -0.214R | REJECTED / CLOSE THE LINE |
| `eurusd_am_post_news_external_liquidity_shift` | BACKTESTED | Post-news external liquidity raid + structural break (M3) | 14 Trades, PF inf, Exp. 0.135R | STRUCTURALLY-PROMISING / LOW_FREQ |

## Current Practical Interpretation

- No hay estrategia viva promotable hoy (alpha).
- `SMT con DXY` no es utilizable hoy por falta de fuente local auditable.
- `08:00 NY` HABILITADO para investigacion cuantitativa.
- La compuerta `AM final` ha sido APROBADA y el laboratorio es bisesion (AM+PM).
- el baseline horario profesional es `08:00-16:30 NY`.
- la familia `Silver Bullet NY AM` ya fue falsada.
- la familia `eurusd_am_post_news_external_liquidity_shift` tiene economia quirúrgica (PF inf) pero frecuencia extrema baja (14 trades / 6 años).
- La decisión final es `STRUCTURALLY-PROMISING-BUT-NOT-PROMOTABLE` (Frecuencia insuficiente).
- El pivot estructural principal recomendado sigue siendo `instrument-first, USDJPY-first`.
- No queda hoy una linea `EURUSD` promotable al 100%.


