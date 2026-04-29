# PHASE37ZI-B ORDER_CHECK READINESS AUDIT

- state: BLOCKED_AUTOTRADING_DISABLED
- order_check_executed: True
- order_check_pass: True
- retcode: 0
- comment: Done
- margin_required: 1945.47
- order_send_executed: False
- account_trade_allowed: True
- terminal_trade_allowed: False
- tradeapi_disabled: True
- conclusion: AUTOTRADING_API_BLOQUEADO_POR_TERMINAL
- action_required: Opciones MT5: desmarcar bloqueo Python API

## Request usado
```json
{
  "action": 1,
  "symbol": "EURUSD",
  "volume": 0.5,
  "type": 0,
  "price": 1.16728,
  "sl": 1.16628,
  "tp": 1.16868,
  "deviation": 20,
  "magic": 370037,
  "comment": "PHASE37ZH_ORDER_CHECK_ONLY",
  "type_time": 0,
  "type_filling": 1
}
```