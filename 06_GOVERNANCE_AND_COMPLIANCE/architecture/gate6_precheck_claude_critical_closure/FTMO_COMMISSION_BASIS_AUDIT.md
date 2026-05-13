# FTMO COMMISSION BASIS AUDIT

## Basis: ROUND-TURN
- FTMO charges USD 5/lot round-turn for EURUSD
- Field name: `commission_per_lot_round_turn` (explicit)
- Formula: `commission_r = commission_usd_per_lot_round_turn / (sl_pips * pip_value_per_standard_lot_usd)`

## Test Case
- EURUSD, 1 lot, SL 10 pips, pip_value = 10 USD/pip
- risk_usd = 100 USD
- commission = 5 USD round-turn
- commission_r = 5 / (10 * 10) = 0.05R

## Per-Side Comparison
- If per-side 5 USD, round-turn = 10 USD → commission_r = 0.10R
- Default is round-turn (confirmed by field name + test)

## Parameter Changeability
- `commission_per_lot_round_turn` is a config field, changeable without touching logic
