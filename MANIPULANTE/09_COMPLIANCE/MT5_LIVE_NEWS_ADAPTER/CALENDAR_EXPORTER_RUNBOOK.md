# Calendar Exporter Runbook

Purpose: generate verified EUR/USD HIGH-impact news cache from the MT5/MQL5 Economic Calendar.

Operational sequence:

1. Confirm MT5 server is `FTMO-Demo`.
2. Confirm account is demo/trial.
3. Run `MANIPULANTE_CalendarExporter.mq5`.
4. Verify generated JSON includes:
   - `source_type = MT5_MQL5_ECONOMIC_CALENDAR`
   - `verified_by_mt5 = true`
   - `generated_at_utc`
   - EUR/USD currencies
   - HIGH impact
   - event time
   - timezone basis
5. Run Phase37B again.

If files are missing, stale, malformed, manual-fake, or timezone ambiguous: `NO_TRADE`.
