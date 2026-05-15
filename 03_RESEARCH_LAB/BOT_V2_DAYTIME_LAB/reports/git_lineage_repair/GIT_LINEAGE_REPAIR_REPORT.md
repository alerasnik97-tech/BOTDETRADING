# GIT LINEAGE REPAIR REPORT
**Fecha**: 2026-05-15
**Status**: **REPAIRED_LOCAL_PENDING_REMOTE**

## 1. Objective
Resolver el problema de historia no relacionada entre la rama de reconciliación y `main`, habilitando la capacidad de abrir Pull Requests reales.

## 2. Lineage Details
- **Base Commit (main)**: 42 commits ahead of origin (local main used as base).
- **New Branch**: `research/v50b-cost-hardening-foundation-20260515`
- **Source Branch**: `research/v50b-evidence-reconciliation-20260515`

## 3. Imported Evidence (Allowlist)
Se importaron únicamente los archivos certificados de la fase de reconciliación, evitando el arrastre de objetos pesados o historias corruptas:
- `EVIDENCE_RECONCILIATION_REPORT.md`
- `FULL_RERUN_TRAIN_ONLY_REPORT.md` (Versión Corregida)
- `QUARANTINED_DO_NOT_USE.md`
- `scripts/utils/integrity.py`
- `tests/remediation/test_integrity.py`
- `MANIFEST_68fa2280.json`
- `V50B_RERUN_MASTER_RANKING.csv` (Canonical Source of Truth)
- `V50B_RERUN_TOP20_GLOBAL.csv`

## 4. PR Compatibility
La nueva rama ahora comparte historia con `main`. Se ha verificado localmente. El push a GitHub está en proceso de re-intento.

## 5. Decision
La fundación de evidencia está restaurada sobre una rama limpia. El laboratorio es ahora auditable profesionalmente.
