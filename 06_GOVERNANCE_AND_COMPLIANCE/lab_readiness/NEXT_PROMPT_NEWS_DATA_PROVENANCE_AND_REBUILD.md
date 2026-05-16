# NEXT PROMPT: NEWS DATA PROVENANCE AND REBUILD

Actua como auditor institucional de news data, provenance engineer y leakage-prevention officer.

Objetivo:
Resolver la fase separada de noticias sin tocar el laboratorio EURUSD core ni activar news para operar.

Reglas:

- No backtest.
- No strategy run.
- No optimization.
- No validation.
- No holdout.
- No usar 2025/2026 para analisis.
- No scraping ni descarga salvo autorizacion explicita del owner.
- No stubs.
- No datos sinteticos.
- No usar `canonical_anchor_events.csv` como autoridad hasta auditar provenance.

Inputs esperados:

- Owner-supplied `forex_factory_cache.csv`, o autorizacion explicita de regeneracion.
- Owner-supplied `news_eurusd_v2_utc.csv`, o autorizacion explicita de rebuild.
- Sidecars de provenance: fuente, fecha, hash, rowcount, schema, timezone, rango temporal, generador usado.

Tareas:

1. Confirmar existencia y hashes de archivos owner-supplied.
2. Auditar schema, timestamps, timezone y duplicados.
3. Auditar provenance de `05_MARKET_DATA_VAULT/data/official_anchors/out/canonical_anchor_events.csv`.
4. Rechazar cualquier fila/proceso con provenance no auditable.
5. Mantener `NewsConfig().enabled == False` hasta cierre de auditoria.
6. Crear tests de news que fallen cerrado si faltan datos reales.
7. Documentar si EURUSD core puede seguir con news disabled.

Decision final requerida:

- NEWS_DATA_PROVENANCE_APPROVED
- NEWS_DATA_BLOCKED_OWNER_FILE_REQUIRED
- NEWS_DATA_BLOCKED_REGEN_AUTHORIZATION_REQUIRED
- NEWS_DATA_REJECTED_PROVENANCE_UNVERIFIED
