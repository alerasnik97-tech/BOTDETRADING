# SLIPPAGE MODEL AUDIT

## Implementation
- `CostModelConfig.slippage_pips` field exists
- Formula: `slippage_r = slippage_pips / sl_pips`
- Integration: `net_r = gross_r - commission_r - slippage_r`
- Applied to ALL exits (TP, SL, BE, TIME, EOM)

## Gate 6 Mini Stress Levels
- Base: 0.0 pip
- Stress: 0.1, 0.2, 0.5 pip

## Monotonicity
- Confirmed by test: increasing slippage monotonically reduces net_r
