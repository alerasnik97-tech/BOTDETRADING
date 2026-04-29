# Install Calendar Exporter

Use this only on the FTMO Demo / Free Trial MT5 terminal.

1. Open MT5 FTMO Demo manually.
2. Use `File > Open Data Folder`.
3. Copy `MANIPULANTE_CalendarExporter.mq5` into `MQL5\Scripts`.
4. Open MetaEditor from MT5.
5. Compile the script.
6. Run it manually on EURUSD.
7. Confirm it only exports calendar data.
8. Confirm it does not contain trading functions.
9. Copy or configure outputs into:
   `MANIPULANTE\09_COMPLIANCE\live_news_cache\`
10. Required Phase37B files:
   - `YYYY-MM-DD_ftmo_news_today.json`
   - `YYYY-MM-DD_ftmo_news_week.json`
   - `YYYY-MM-DD_ftmo_news_gate_status.json`

Do not enable real trading. Do not use this to send orders.
