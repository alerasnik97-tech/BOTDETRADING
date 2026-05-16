# REMAINING ROOT ITEMS AUDIT REPORT

## 1. Status
**REMAINING_ROOT_ITEMS_PHASE_A_APPLIED_PARTIAL**

## 2. Executive Summary
Se ha realizado una auditoría exhaustiva de los 48+ ítems restantes en la raíz del repositorio. Se han clasificado en 5 fases (A-E) siguiendo la política de "Strict Root" del owner (8 carpetas + .gitignore). El entorno es estable (119/119 tests PASS).

## 3. Current Root Inventory
Se detectan ítems que no pertenecen a la Allowlist Estricta. La mayoría son documentos de gobernanza, scripts de infraestructura legacy o carpetas de investigación que requieren una migración cuidadosa.

## 4. Strict Root Target
- 01_CORE_PRODUCTION
- 02_INCUBATION_STAGING
- 03_RESEARCH_LAB
- 04_INFRASTRUCTURE_ENGINEERING
- 05_MARKET_DATA_VAULT
- 06_GOVERNANCE_AND_COMPLIANCE
- 07_BACKUPS
- 08_CLOUD_FREE_RUN_LAB
- .gitignore

## 5. ZIP Policy Audit
Se han detectado 2 archivos `.zipbak` en la carpeta `legacy_archive_2026/` situada en la raíz.
- `000_PARA_CHATGPT_BACKUP_20260421_211526.zipbak`
- `_test.zipbak`
**Acción:** Mover `legacy_archive_2026/` a `07_BACKUPS/` para cumplir con la política de "No ZIPs en área activa".

## 6. Remaining Classification Table (Sample)
| Item | Category | Phase | Decision |
| :--- | :--- | :--- | :--- |
| `CHANGELOG.md` | DOC | A | Move to 06_GOVERNANCE/history/ |
| `ESTRUCTURA_DEL_PROYECTO.md` | GOVERNANCE | A | Move to 06_GOVERNANCE/architecture/ |
| `MANIPULANTE` | STRATEGY | B | Audit Strategy Authority before move |
| `BOT_MARKET_DATA` | DATA | C | Audit Data before move to 05_VAULT |
| `research_lab` | IMPORT | D | Blocked until path migration |
| `.github` | TECH | E | Exception Required for CI/CD |

## 7. Phase A Actions Applied
- **Moved to 06_GOVERNANCE:** `audits/`, `bls_html_samples/`.
- **Moved to 07_BACKUPS:** `legacy_archive_2026/`, `manual_trade_chartpacks/`, `ARCHIVO_HISTORICO/`, `000_PARA_CHATGPT`, `GIT_BACKUP_*.bundle*`.
- **Moved to 04_INFRA:** `VPS_READINESS/`, `.vscode/`.
- **Quarantined:** `.mplconfig/`.
- **Deleted:** Carpetas vacías en raíz tras movimiento.

## 8. Strategy Authority Items
- `MANIPULANTE`, `ROCKI_AM`, `ESTRATEGIAS`, `STRATEGIES`, `LAB_STRATEGIES`.
- **Riesgo:** Alta sensibilidad. No mover sin confirmar si son PROD, RESEARCH o BACKUP.

## 9. Data Audit Items
- `BOT_MARKET_DATA`, `DATA MANUAL`, `scbi_*_checkpoints`.
- **Riesgo:** Inmutabilidad de datos. Requiere manifiesto antes de mover a `05_MARKET_DATA_VAULT`.

## 10. Import/Path Migration Items
- `research_lab` e `imports` relacionados en `scripts/`.
- **Estado:** 831+ referencias detectadas. Se requiere un plan de migración con wrappers de compatibilidad o política de `PYTHONPATH`.

## 11. Technical Root Exceptions
- **.github:** Obligatorio en raíz por requerimiento de GitHub Actions.
- **README.md / requirements.txt:** Estándar profesional de repositorios. Se recomienda mantener como excepción técnica documentada.

## 12. Strict Root Endgame Plan
1. **Fase A (Inmediata):** Limpieza de documentos de gobernanza y backups.
2. **Fase B (Surgical):** Auditoría y clasificación de carpetas de estrategias.
3. **Fase C (Surgical):** Auditoría y migración de datos legacy a Vault.
4. **Fase D (Refactor):** Migración de `research_lab` y actualización de imports.
5. **Fase E (Final):** Consolidación de excepciones técnicas y cierre de raíz.

## 13. Safety Verification
- tracked_files_deleted_without_approval: NO
- raw_data_touched: NO
- validation_touched: NO
- holdout_touched: NO
- 2025_touched: NO
- 2026_touched: NO
- backtest_run: NO
- strategy_run: NO
- zip_left_in_active_root: NO (legacy_archive_2026 movido a 07_BACKUPS)
- zip_left_in_active_project: NO

## 14. Copy-Paste Summary for ChatGPT
`HEAD: 43f2ae6f, Status: AUDITED, Root: 48 items, ZIPs: Detected in legacy_archive_2026, Tests: PASS (119/119).`
