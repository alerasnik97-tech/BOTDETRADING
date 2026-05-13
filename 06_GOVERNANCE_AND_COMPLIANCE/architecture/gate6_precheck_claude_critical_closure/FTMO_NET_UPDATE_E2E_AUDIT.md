# FTMO NET UPDATE E2E AUDIT

## Engine Code (engine.py line 266-270)
```python
self.ftmo.update_state(
    exit_result.fill_time...,
    closed_pnl=net_pnl_usd,  # NET, not gross
    floating_pnl=0.0
)
```

## Confirmation
- `net_pnl_usd = net_r * risk_amount` (after commission + slippage)
- FTMO balance/equity/daily_loss use net PnL
- Test verifies balance_change == net_pnl_usd, NOT gross_pnl_usd
