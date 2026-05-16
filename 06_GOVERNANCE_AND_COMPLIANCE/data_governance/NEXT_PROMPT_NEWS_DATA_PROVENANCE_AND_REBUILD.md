# NEXT PROMPT: NEWS DATA PROVENANCE AND REBUILD

Actua como auditor institucional de news data, provenance engineer y leakage-prevention officer.

Objetivo: resolver la fase separada de noticias sin tocar el laboratorio EURUSD core ni activar news para operar.

Reglas:
- No backtest.
- No strategy run.
- No optimization.
- No validation.
- No holdout research.
- No usar 2025/2026 para analisis de estrategia.
- No scraping ni descarga salvo autorizacion explicita del owner.
- No stubs.
- No datos sinteticos.
- No usar `canonical_anchor_events.csv` como autoridad hasta auditar provenance.

Inputs esperados:
- Owner-supplied `forex_factory_cache.csv`, o autorizacion explicita de regeneracion.
- Owner-supplied `news_eurusd_v2_utc.csv`, o autorizacion explicita de rebuild.
- Sidecars de provenance: fuente, fecha, hash, rowcount, schema, timezone, rango temporal, generador usado.

Decision final requerida:
- NEWS_DATA_PROVENANCE_APPROVED
- NEWS_DATA_BLOCKED_OWNER_FILE_REQUIRED
- NEWS_DATA_BLOCKED_REGEN_AUTHORIZATION_REQUIRED
- NEWS_DATA_REJECTED_PROVENANCE_UNVERIFIED
