# PHASE50C-E TRADE 2316 TICK GAP AUDIT

Verdict: TRADE_2316_EXTRACTION_GAP_REQUIRES_PATCH

Trade 2316:
- entry/exit NY: 2025-01-14T07:45:00-05:00 / 2025-01-14T08:33:00-05:00
- Phase50C-D window: 2025-01-14T07:35:00-05:00 to 2025-01-14T08:43:00-05:00
- bar/tick: BE / NONE

Tick availability:
- total_ticks_day: 46291
- ticks_0745_0833_ny: 0
- ticks_0745_0833_utc_equivalent: 0
- max_gap_seconds: 28808.808

Cache:
- M1 rows/tick_count: 0 / None
- M5 rows/tick_count: 0 / None
- M15 rows/tick_count: 0 / None

Redownload:
- executed: True
- rows_redownloaded: 10551
- rows_canonical_same_window: 0
- state: EXTRACTION_GAP_REQUIRES_PATCH

Safety:
- MANIPULANTE not modified.
- Strategy, MT5, orders, real, Exness, Git add/commit/push not touched.
