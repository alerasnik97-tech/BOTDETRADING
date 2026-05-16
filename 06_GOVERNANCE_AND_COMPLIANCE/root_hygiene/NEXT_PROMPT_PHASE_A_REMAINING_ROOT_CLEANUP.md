# NEXT PROMPT: PHASE A REMAINING ROOT CLEANUP

## Objetivo
Ejecutar la limpieza quirúrgica de ítems de gobernanza, documentos y backups identificados como Fase A (bajo riesgo, sin código/datos/estrategias).

## Acciones Phase A Propuestas
1. **Move Governance to 06_GOVERNANCE:**
   - `01_CURRENT_PROJECT_STATUS.*` -> `06_GOVERNANCE_AND_COMPLIANCE/status/`
   - `02_STRATEGY_AUTHORITY_MAP.*` -> `06_GOVERNANCE_AND_COMPLIANCE/strategy_authority/`
   - `03_OBSOLETE_AND_SUPERSEDED_INDEX.*` -> `06_GOVERNANCE_AND_COMPLIANCE/obsolete/`
   - `CHANGELOG.md` -> `06_GOVERNANCE_AND_COMPLIANCE/history/`
   - `ESTRUCTURA_DEL_PROYECTO.md` -> `06_GOVERNANCE_AND_COMPLIANCE/architecture/`
   - `AUDITORIA_EJECUCION_FINAL.md` -> `06_GOVERNANCE_AND_COMPLIANCE/audits/`

2. **Move Backups to 07_BACKUPS:**
   - `legacy_archive_2026/` -> `07_BACKUPS/legacy_archive_2026/` (CUMPLE ZIP POLICY)
   - `manual_trade_chartpacks/` -> `07_BACKUPS/manual_trade_chartpacks/`
   - `ARCHIVO_HISTORICO/` -> `07_BACKUPS/ARCHIVO_HISTORICO/`
   - `GIT_BACKUP_20260507_*.bundle*` -> `07_BACKUPS/git_bundles/`

3. **Move Infrastructure/Lab Docs:**
   - `VPS_READINESS/` -> `04_INFRASTRUCTURE_ENGINEERING/vps/`
   - `CLOUD_WORKFLOW.md` -> `08_CLOUD_FREE_RUN_LAB/docs/`
   - `CANONICAL_EXECUTION_CONTRACT.md` -> `06_GOVERNANCE_AND_COMPLIANCE/protocols/`

## Verificación
- `git status`
- `python -c "import research_lab"`
- `python -m unittest discover ...` (F06 tests)
