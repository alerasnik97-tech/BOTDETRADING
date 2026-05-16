# NEWS DATA FOUNDATION SCOPE DECISION

## Status

NEWS_SCOPE_SEPARATE_FAIL_CLOSED

## Decision

News data is not enabled for EURUSD core prepared OHLCV loading and is not used in this phase.

## Evidence

- `NewsConfig().enabled` is `False`.
- `05_MARKET_DATA_VAULT/data/forex_factory_cache.csv` is missing.
- `05_MARKET_DATA_VAULT/data/news_eurusd_v2_utc.csv` is missing.
- `05_MARKET_DATA_VAULT/data/official_anchors/out/canonical_anchor_events.csv` is present but provenance remains unverified.

## Scope Boundary

News rebuild/provenance is a separate phase. It must not use fake stubs, scraping, downloads, or 2025/2026 strategy analysis without explicit owner authorization.
