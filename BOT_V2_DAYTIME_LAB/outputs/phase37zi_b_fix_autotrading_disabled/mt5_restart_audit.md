# PHASE37ZI-B MT5 RESTART AUDIT

- verdict: TERMINAL_API_STILL_DISABLED
- runner_pids_before_stop: [17376]
- runner_pids_after_stop: []
- runner_pids_after_restart: [9520]

## state_before
- server: FTMO-Demo
- trade_mode: 0
- account_trade_allowed: True
- terminal_trade_allowed: False
- tradeapi_disabled: True
- positions_total: 0

## state_after_mt5_restart
- server: FTMO-Demo
- trade_mode: 0
- account_trade_allowed: True
- terminal_trade_allowed: False
- tradeapi_disabled: True
- positions_total: 0

## state_after_runner_restart
- server: FTMO-Demo
- trade_mode: 0
- account_trade_allowed: True
- terminal_trade_allowed: False
- tradeapi_disabled: True
- positions_total: 0