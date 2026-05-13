# MANIPULANTE 4 — FEATURE SPECIFICATION

## Per-Sweep Candidate Features

### Identity
- sweep_id: unique identifier
- date: YYYY-MM-DD
- session: NY_AM
- direction: LONG/SHORT
- level_type: PDH_PDL / ASIA_HL / PWH_PWL / H1_SWING
- level_price: float
- sweep_time: datetime UTC
- reclaim_time: datetime UTC (or NaN if no reclaim)

### Sweep Quality Metrics
- sweep_depth_pips: how far price went beyond the level
- sweep_depth_atr_ratio: sweep_depth / ATR_H1_20
- sweep_duration_seconds: time from breach to max excursion
- reclaim_occurred: True/False
- reclaim_time_seconds: seconds from breach to close back inside level
- close_back_inside: True/False (did price CLOSE a bar back inside)
- wick_body_ratio: wick / total range of the sweep bar
- rejection_candle_range_atr: range of rejection bar / ATR
- post_sweep_adverse_extension_pips: max further extension after sweep before reclaim

### Displacement Metrics
- displacement_exists: True/False
- displacement_time: datetime UTC
- displacement_direction: matches expected reversal direction
- displacement_body_atr_ratio: displacement candle body / ATR
- displacement_range_atr_ratio: displacement candle range / ATR
- displacement_close_strength: 0.0-1.0 (close position within range)
- displacement_breaks_ltf_structure: True/False
- displacement_creates_fvg: True/False
- displacement_speed_seconds: time to complete displacement sequence

### Context
- htf_bias_type: D0_NO_BIAS / D3_PREMIUM_DISCOUNT
- distance_to_opposing_liquidity_pips: float
- atr_regime: LOW/NORMAL/HIGH (percentile-based)
- spread_at_sweep: current spread in pips
- time_of_day_ny: HH:MM
- news_context: CLEAR / WITHIN_BUFFER / POST_TIER1
- rollover_blocked: True/False

### Execution (post-fill)
- entry_type: STOP_CONFIRMATION / FVG_50PCT
- entry_time: datetime UTC
- entry_price: float
- sl_price: float
- tp_price: float
- tp_r: 2.0 / 2.5
- be_rule: NONE / 1.25R
- risk_pips: float
- commission_r: float (dynamic by sl_pips)
- slippage_r: float
- gross_r: float
- net_r: float
- exit_reason: TP / SL / BE / TIME / EOM
- eom_type: NO_EOM / NATURAL / ARTIFICIAL
