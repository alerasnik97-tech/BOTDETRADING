# PHASE44 TEST RESULTS

- timestamp_utc: 2026-04-30T11:46:56.174001+00:00
- total: 11
- pass: 11
- fail: 0
- warnings: 1

## 1. Crear/compilar scripts Phase44
- pass: True
- evidence: ``

## 2. Crear SQLite
- pass: True
- evidence: `db=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\db\manipulante_observability.sqlite; tables=['bot_heartbeats', 'daily_scorecards', 'decisions', 'fills_manual', 'incidents', 'sqlite_sequence']; size=339968`

## 3. Ingestar logs actuales
- pass: True
- evidence: `counts={'bot_heartbeats': 4, 'decisions': 673, 'daily_scorecards': 1, 'incidents': 2, 'fills_manual': 0}; invalid_legacy_rows_filtered_in_views=372`
- warning: Hay filas antiguas mal mapeadas preservadas en SQLite, filtradas por health/dashboard.

## 4. Generar health snapshot
- pass: True
- evidence: `health=HEALTH_BLOCKED_BY_RULE; path=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\daily\latest_health_snapshot.json`

## 5. Generar daily report
- pass: True
- evidence: `verdict=OBS_DAY_CLEAN_NO_TRADE; path=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\daily\2026-04-30_daily_observability_report.json`

## 6. Crear dashboard o fallback HTML
- pass: True
- evidence: `streamlit_available=True; html=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\dashboard\dashboard.html; size=9419`

## 7. Confirmar no toca MT5
- pass: True
- evidence: `No imports/calls to MetaTrader5 found in Phase44 scripts.`

## 8. Confirmar no envia ordenes
- pass: True
- evidence: `No order_send call found in Phase44 scripts.`

## 9. Confirmar no modifica estrategia/runner/START/STATUS/STOP
- pass: True
- evidence: `strategy_touched=[]`

## 10. Confirmar no guarda secretos
- pass: True
- evidence: `secret_hits=[]`

## 11. Confirmar tolera logs faltantes
- pass: True
- evidence: `missing_log_helpers_ok`
