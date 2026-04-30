# Phase42C Diagnostic

- timestamp_utc: 2026-04-30T10:55:51.031532+00:00
- root: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
- STOP_BOT exists: True
- STOP_BOT path: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STOP_BOT.txt`
- runner active: False
- runner count: 0
- runner pids: []
- position open: False
- FTMO Demo: True
- Exness detected: False
- Real detected: False
- AutoTrading OK: True
- STATUS actual: BOT DETENIDO
- boundary preflight exit_code: 2

## Notes
- START preflight is read-only until the BAT clears STOP_BOT after all gates pass.
- No order_send was executed by this diagnostic.
- The legacy boundary preflight reported missing old critical files; this task stayed inside the canonical root and only touched the requested MANIPULANTE/Phase42C surfaces.
