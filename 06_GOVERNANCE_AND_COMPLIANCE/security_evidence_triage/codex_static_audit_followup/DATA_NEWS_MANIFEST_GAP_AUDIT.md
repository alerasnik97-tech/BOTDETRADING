# DATA / NEWS MANIFEST GAP AUDIT

Status: COMPLETE_STATIC_READ
Decision class: BLOCKS_PAPER / BLOCKS_DEMO_FUNDED

## Physical inventory observed
- Tick vault exists at `05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD/monthly`.
- Parquet count observed: 137 files, from `EURUSD_ticks_2015_01.parquet` through `EURUSD_ticks_2026_04.parquet` plus pilot file.
- News file exists: `05_MARKET_DATA_VAULT/data/news_eurusd_am_fortress_v3.csv`.
- Restore manifest exists: `05_MARKET_DATA_VAULT/data/NEWS_RESTORE_MANIFEST.json`.

## Confirmed governance evidence
- `06_GOVERNANCE_AND_COMPLIANCE/data_quality_audits/parallel_data_news_audit/01_MARKET_DATA_COVERAGE_AUDIT.md` claims contiguous 2015-01 to 2026-04 coverage.
- `NEWS_CALENDAR_COVERAGE_BY_MONTH.csv`, `EURUSD_TICK_COVERAGE_BY_MONTH.csv`, `EURUSD_SPREAD_QUALITY_BY_MONTH.csv`, and `EURUSD_TIMESTAMP_QUALITY_BY_MONTH.csv` exist.
- `06_GOVERNANCE_AND_COMPLIANCE/data_news_vault_integrity/v49_7_parallel_audit/NEWS_FILE_HASH_AUDIT.csv` exists.

## Findings

### DATA-001 - Vault README references canonical files that are physically missing
- Severity: MEDIUM
- Status: CONFIRMED_ACTIVE
- Evidence: `05_MARKET_DATA_VAULT/README.md` references `DATA_MANIFEST.csv` and `SCHEMA.md`; both are absent in `05_MARKET_DATA_VAULT`.
- Impact: secondary governance audits exist, but the vault itself does not carry the promised canonical manifest/schema.
- Classification: BLOCKS_PAPER, BLOCKS_DEMO_FUNDED; not alone a V50B research blocker.
- Required correction: create canonical vault-local manifest/schema or update README to point to the governance audit source of truth.

### DATA-002 - Data/news audits are not integrated as a hard gate in current V50B evidence
- Severity: HIGH
- Status: CONFIRMED_ACTIVE
- Evidence: current V50B runner does not consume real market data or news data.
- Impact: V50B evidence cannot inherit the quality audits.
- Classification: BLOCKS_CURRENT_RESEARCH
- Required correction: rebuild V50B evidence from real data paths and explicitly attach data/news audit references.

### NEWS-001 - News coverage exists but must be fail-closed at runner boundary
- Severity: MEDIUM
- Status: CONFIRMED_ACTIVE
- Evidence: governance reports mark AM Fortress v3 as primary; runner-level enforcement must still be proven per family.
- Impact: current static audit cannot certify that every future runner will fail closed on missing news.
- Classification: BLOCKS_PAPER, BLOCKS_DEMO_FUNDED
- Required correction: require per-run coverage proof and explicit missing-news failure state.
