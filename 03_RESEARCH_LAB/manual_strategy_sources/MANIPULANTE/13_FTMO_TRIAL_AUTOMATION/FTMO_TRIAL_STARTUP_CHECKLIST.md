# FTMO Trial Startup Checklist

1. Confirm account is FTMO demo/trial.
2. Run `MANIPULANTE_CalendarExporter.mq5`.
3. Confirm `*_ftmo_news_today.json` exists.
4. Confirm `*_ftmo_news_week.json` exists.
5. Confirm News Gate = ALLOW.
6. Confirm live Signal Sync = OK.
7. Confirm Data/Time/Symbol/Lot Gates = ALLOW.
8. Dry-run first.
9. Only after all gates pass, create confirmation file manually.
10. Only after all gates pass, remove STOP_BOT intentionally.
