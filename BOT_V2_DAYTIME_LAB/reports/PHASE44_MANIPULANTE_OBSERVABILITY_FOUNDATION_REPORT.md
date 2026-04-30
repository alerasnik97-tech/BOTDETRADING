# PHASE44 MANIPULANTE OBSERVABILITY FOUNDATION REPORT

- timestamp_utc: 2026-04-30T11:51:44.464304+00:00
- verdict: OBSERVABILITY_FOUNDATION_READY_WITH_WARNINGS

## 1. Lo mas importante
Se creo una capa local read-only de observabilidad para MANIPULANTE: SQLite, JSONL, health snapshot, daily report y dashboard local. No toca MT5, no envia ordenes y no modifica estrategia.

## 2. Observability creada
- carpeta: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY`
- SQLite: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\db\manipulante_observability.sqlite` (339968 bytes, local-only)
- JSONL: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\jsonl\bot_events.jsonl` (595210 bytes, local-only)
- health snapshot: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\daily\latest_health_snapshot.md`
- db_counts: `{'bot_heartbeats': 4, 'decisions': 673, 'daily_scorecards': 1, 'incidents': 2, 'fills_manual': 0}`

## 3. Dashboard
- Streamlit disponible: True
- HTML fallback: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\dashboard\dashboard.html`
- BAT: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\dashboard\ABRIR_DASHBOARD_MANIPULANTE.bat`

## 4. Daily report
- creado: True
- ruta: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\daily\2026-04-30_daily_observability_report.md`
- veredicto: OBS_DAY_CLEAN_NO_TRADE

## 5. Datos mostrados
- bot status, noticias, ordenes, decisiones, scorecard, incidents, promotion gate, stress tests y fills manuales si existen.

## 6. Seguridad
- no MT5: true
- no ordenes: true
- no real: true
- no Exness: true
- no estrategia modificada: true
- no secretos: true

## 7. GitHub policy
- subir: scripts, docs, dashboard liviano, reportes md/json de fase, daily md, ZIP canonico si liviano.
- no subir: SQLite local, JSONL local, daily JSON operativo, logs pesados, secretos, credenciales, MT5 account files.

## 8. Tests
- total: 11
- pass: 11
- fail: 0
- warnings: 1

## 9. ZIP
- Validacion fisica en `BOT_V2_DAYTIME_LAB/outputs/phase44_manipulante_observability_foundation/phase44_zip_validation.*`.

## 10. Warnings
- Hay filas antiguas mal mapeadas preservadas en SQLite, filtradas por health/dashboard.
