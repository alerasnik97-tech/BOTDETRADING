# CHANGELOG — BOT DE TRADING

Formato: [YYYY-MM-DD] Categoría: Descripción

---

## [Unreleased]

*Cambios pendientes de próximo commit/release.*

---

## [2026-04-14] — Infrastructure Hardening Complete

### Added
- `run_canonical.py`: entrypoint único y oficial con guardrails estrictos
- `research_lab/rejection_protocol.py`: protocolo de rechazo IS/OOS formal
- `research_lab/version.py`: sistema de versionado del laboratorio
- `research_lab/tests/test_rejection_harness.py`: tests de umbrales IS/OOS
- `research_lab/tests/test_e2e_canonical_flow.py`: smoke test canónico E2E
- `STRATEGY_PROMOTION_POLICY.md`: taxonomía formal HARD_REJECT → LIVE_CANDIDATE
- `CANONICAL_EXECUTION_CONTRACT.md`: contrato de ejecución y outputs
- `COMPARABILITY_2020_2025_NOTE.md`: veredicto formal de comparabilidad temporal
- `INFRASTRUCTURE_STATUS_FINAL.md`: estado sellado de infraestructura
- `lineage_metadata.json`: generado por cada corrida, inmutable, trazable

### Changed
- `research_lab/main.py`: inyección de `final_promotion_status` y versiones en lineage
- `research_lab/news_filter.py`: corrección `DEFAULT_NEWS_FILE` → `DEFAULT_NEWS_V2_UTC_FILE`
- `research_lab/config.py`: STRATEGY_NAMES extendido con familia Sprint 2

### Fixed
- `research_lab/main.py`: restaurado `DEFAULT_DATA_DIRS` en import (stale desde sesión anterior)
- `research_lab/main.py`: guard `if "default_wfa" not in item: continue` para estrategias abortadas IS

---

## [2026-04-13] — News Pipeline V2 UTC + Discovery Sprint 1

### Added
- `research_lab/news_tradingeconomics.py`: nuevo parser UTC canónico para noticias
- `research_lab/news_phase3_mass_validate.py`: validación masiva del pipeline de noticias
- `research_lab/strategies/strategy_ny_br_ema.py`: estrategia familia breakout+retest
- `research_lab/strategies/strategy_ny_br_pure.py`: variante sin filtro EMA
- `research_lab/strategies/strategy_ny_br_mom.py`: variante con momentum
- `docs/SOURCE_INTEGRATION.md`: documentación del proceso de integración de fuentes

### Changed
- `research_lab/data_loader.py`: inyección de columnas AM Range (07:00–11:00 NY)

---

## [2026-04-12] — Purge Controlado + Confinamiento Legacy

### Added
- `legacy_archive_2026/`: confinamiento de scripts temporales y legacy root
- `MIGRATION_MANIFEST_W2.md`: manifiesto de migración semana 2
- `LEGACY_CONFINEMENT_REPORT_W2.md`: reporte de confinamiento

### Changed
- Root del repo significativamente limpiado

---

## [2026-04-08] — Base de Laboratorio Canónico

### Added
- `research_lab/`: motor completo de backtesting (engine, validation, scorer, report)
- `research_lab/strategies/`: familias base (donchian, bollinger, EMA, etc.)
- `requirements.txt`
- Estructura inicial de `data_free_2020/` y `data_candidates_2022_2025/`

### Changed
- Migración desde versión escritorio a `C:\BOT DE TRADING` como fuente oficial

---

## [Histórico anterior]

*Ver commits de Git para historial completo pre-abril 2026.*
