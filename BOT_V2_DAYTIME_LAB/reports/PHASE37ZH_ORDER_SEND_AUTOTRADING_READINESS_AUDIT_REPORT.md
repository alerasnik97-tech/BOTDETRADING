# PHASE37ZH ORDER_SEND AUTOTRADING READINESS AUDIT REPORT

## 1. Lo mas importante

MT5 no esta listo para ejecutar ordenes demo aunque la cuenta FTMO Demo este confirmada.

La cuenta permite trading, pero el terminal tiene AutoTrading/API bloqueado:

- `account.trade_allowed`: `true`
- `account.trade_expert`: `true`
- `terminal.trade_allowed`: `false`
- `terminal.tradeapi_disabled`: `true`

Se ejecuto solo `mt5.order_check`.

No se ejecuto `mt5.order_send`.

STATUS final:

`ESTADO GENERAL: BLOQUEADO - AUTOTRADING DESHABILITADO`

## 2. Veredicto final exacto

BLOCKED_AUTOTRADING_DISABLED

## 3. Cuenta

- FTMO Demo confirmado: `SI`
- Company: `FTMO Global Markets Ltd`
- Server: `FTMO-Demo`
- Modo: `DEMO`
- Balance: `10000.0`
- Currency: `USD`
- Exness detectado: `NO`
- Real detectado: `NO`

## 4. Runner

- PID unico: `12512`
- Duplicados: `NO`
- STATUS no muestra DUPLICADO: `SI`

## 5. AutoTrading / terminal permissions

- `trade_allowed` cuenta: `true`
- `trade_expert` cuenta: `true`
- `terminal_trade_allowed`: `false`
- `tradeapi_disabled`: `true`
- `connected`: `true`
- `dlls_allowed`: `true`

Conclusion:

La cuenta FTMO Demo esta bien, pero MT5 rechazaria una orden real por permisos de terminal/API. El bot debe quedar bloqueado, no OK.

## 6. OrderCheck

- Ejecutado: `SI`
- `order_send` ejecutado: `NO`
- Symbol: `EURUSD`
- Volumen hipotetico: `0.5`
- Riesgo usado para request: `0.50%`
- SL hipotetico: `10 pips`
- TP hipotetico: `14 pips`
- Filling mode: `ORDER_FILLING_IOC`
- Retcode: `0`
- Comment: `Done`
- Pass/fail: `PASS`
- Margen requerido aproximado: `1946.55`

Interpretacion:

`order_check` pasa, pero `terminal_trade_allowed=false` y `tradeapi_disabled=true` bloquean la capacidad real de enviar ordenes. Por eso el estado final correcto es `BLOCKED_AUTOTRADING_DISABLED`.

## 7. STATUS final

- Estado general: `BLOQUEADO - AUTOTRADING DESHABILITADO`
- Bot: `ACTIVO`
- Cuenta: `FTMO-Demo / DEMO`
- Runner: `ACTIVO`
- PID runner: `12512`
- MT5: `ABIERTO`
- Ordenes: `BLOQUEADAS POR MT5`
- ORDER_CHECK: `PASS`
- ORDER_SEND: `GATEADO`
- Accion: `Revisar boton Trading algoritmico en MT5`

## 8. Dry-run

- Comando: `python BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_bot_runner.py --ftmo-trial --dry-run --risk 0.005 --no-real --once`
- Decision: `NO_TRADE_AUTOTRADING_DISABLED`
- Reason: `AUTOTRADING_DESHABILITADO`
- `order_sent`: `false`
- Posicion posterior: `FLAT`

## 9. Seguridad

- No real: `SI`
- No Exness: `SI`
- No estrategia modificada: `SI`
- No TP modificado: `SI`
- No BE modificado: `SI`
- No BF modificado: `SI`
- No orden enviada: `SI`
- No posicion de prueba abierta: `SI`
- No logs borrados: `SI`

## 10. ZIP/Git

ZIP:

- `000_PARA_CHATGPT.zip` actualizado.
- `testzip`: `None`.
- Duplicados internos: `0`.
- Reporte Phase37ZH incluido: `SI`.

Git:

- Commit selectivo: `Phase37ZH order send autotrading readiness audit`.
- Push destino: `origin main`.

## 11. Siguiente paso unico

Activar el boton `Trading algoritmico` en MT5 y volver a ejecutar esta auditoria sin enviar ordenes.
