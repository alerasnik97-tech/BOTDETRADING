# MANIPULANTE 4 — DATA/NEWS INTEGRATION

## Calendar Source
- **Primary:** AM Fortress v3 (`news_eurusd_am_fortress_v3.csv`)
- **Location:** `05_MARKET_DATA_VAULT/data/`
- **Status:** Verified present, 1106+ events loaded

## Periods
- **Primary reliable:** 2020-01 to 2026-04
- **Legacy reserve (2015-2019):** NOT used for M4 micro-probe. Only available as secondary reference.

## Active Filters
- **Rollover block:** 16:55 to 17:15 NY — all signals blocked
- **Tier-1 standard buffer:** -1 min / +5 min around high-impact events
- **Tier-1 FOMC/rates buffer:** -2 min / +10 min
- **Currency filter:** EUR and USD events only
- **Impact filter:** High impact only

## Fail-Close Rules
- Missing calendar file → abort immediately
- Calendar gap > 5 business days → abort immediately
- Unknown timezone in calendar → abort immediately
- No silent default to "allow all"

## Approval Thresholds
- PF_net > 1.15 under slippage 0.2 for VAL approval
- PF_net > 1.00 under slippage 0.2 for TEST confirmation

## Confirmed Compliance
- [x] Calendar file exists and loads
- [x] Rollover blocker active
- [x] Tier-1 buffers configured
- [x] Fail-close enforced in engine
- [x] Legacy period excluded from main probe
