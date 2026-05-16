# BRANCH HYGIENE AUDIT REPORT

## 1. Status
BRANCH_HYGIENE_PLAN_READY_FOR_OWNER_APPROVAL

## 2. Executive Summary
El repositorio `alerasnik97-tech/bottrading` presenta una acumulación de ramas de gobernanza y research que reflejan el intenso trabajo de remediación de los últimos días. La topología de gobernanza es limpia y lineal, pero existen ramas redundantes (v1 vs v2) y ramas de investigación con PRs en estado *Draft* que deben ser preservadas hasta su consolidación. El plan propone la eliminación de ramas de prueba y versiones obsoletas de remediaciones, protegiendo todo el linaje canónico que conduce al laboratorio EURUSD.

## 3. Branch Inventory

| Branch | Head SHA | Date | Category | Action |
| :--- | :--- | :--- | :--- | :--- |
| `main` | `bd32a412` | 2026-05-07 | MAIN_PROTECTED | KEEP |
| `clean-sync-branch` | `ec1c226a` | 2026-05-16 | DONOR_HISTORICAL | ARCHIVE/TAG |
| `governance/engine-base-preflight-fix-v2-20260516` | `ec1c226a` | 2026-05-16 | ACTIVE_WORKING_BRANCH | KEEP (ACTIVE) |
| `governance/engine-base-preflight-fix-20260516` | `47e9868e` | 2026-05-16 | SUPERSEDED_BY_V2 | DELETE |
| `governance/root-strict-final-pass-20260516` | `d0b33245` | 2026-05-16 | CANONICAL_GOVERNANCE | PROTECT |
| `governance/phase-d-reconciliation-20260516` | `3707a0be` | 2026-05-16 | CANONICAL_GOVERNANCE | PROTECT |
| `governance/root-hygiene-20260516` | `e15df044` | 2026-05-16 | CANONICAL_GOVERNANCE | PROTECT |
| `research/f06-d5-behavior-neutral-telemetry-20260516` | `ef89fddc` | 2026-05-16 | ACTIVE_RESEARCH | KEEP (PR #7) |
| `research/f06-clean-train-only-rerun-20260515` | `c1dae887` | 2026-05-16 | ACTIVE_RESEARCH | KEEP (PR #6) |
| `research/f06-evidence-rebuild-foundation-v2-20260515` | `91be854c` | 2026-05-15 | ACTIVE_RESEARCH | KEEP (PR #5) |
| `research/f06-evidence-rebuild-foundation-20260515` | `e62da979` | 2026-05-15 | SUPERSEDED_BY_V2 | DELETE (PR #4) |
| `research/pre-claude-blocker-remediation-20260515` | `73c4ea63` | 2026-05-15 | ACTIVE_RESEARCH | KEEP (PR #3) |
| `research/push-test-20260515` | `69fa4921` | 2026-05-15 | AGENT_TEMP_BRANCH | DELETE |

## 4. Canonical Branch Decision
La rama canónica actual de gobernanza es **`governance/engine-base-preflight-fix-v2-20260516`**. Contiene la remediación del motor base, la guarda de preflight y la limpieza de raíz estricta. Todo el linaje anterior (`root-strict-final-pass`, `phase-d-reconciliation`) está contenido en su historia de commits.

## 5. F06 / Research Branches
Las ramas F06 están organizadas en una cadena de PRs en Draft (#3 -> #5 -> #6 -> #7). Se recomienda mantenerlas todas hasta que se autorice el laboratorio y se decida el merge final a `main`. La rama `research/f06-evidence-rebuild-foundation-20260515` (PR #4) es redundante frente a su versión V2.

## 6. Legacy / Agent Branches
Las ramas bajo el namespace `agent/*` y `v50b/*` contienen evidencia de ejecuciones previas y auditorías externas. No se recomienda borrarlas sin antes crear Tags institucionales (`archive/v50b-reconciliation`, etc.).

## 7. PR Audit
- **PR #7 (Draft)**: Activo. Base para telemetría D5.
- **PR #6 (Draft)**: Activo. Base para el runner F06.
- **PR #5 (Draft)**: Activo. Base para las guardas de evidencia.
- **PR #4 (Draft)**: **SUPERSEDED**. Candidato a cierre.
- **PR #3 (Draft)**: Activo. Base de remediación pre-Claude.

## 8. Recommended Actions
1.  **Tagging**: Etiquetar `clean-sync-branch` como `legacy/sync-20260516-pre-engine-fix`.
2.  **Deletion**: Eliminar ramas superseded locales y remotas tras aprobación.
3.  **Consolidation**: Mantener el chain de PRs activo para el laboratorio.

## 9. Delete Candidates
- `research/f06-evidence-rebuild-foundation-20260515` (Remote/Local)
- `governance/engine-base-preflight-fix-20260516` (Remote/Local)
- `research/push-test-20260515` (Remote/Local)

## 10. Protected Branches
- `main`
- `governance/engine-base-preflight-fix-v2-20260516`
- Todas las ramas con PRs activos no marcados como SUPERSEDED.

## 11. Risks
- El borrado de `clean-sync-branch` sin tag podría dificultar la comparación con el estado "pre-fix" si surgieran regresiones no detectadas en los tests actuales.

## 12. Copy-Paste Summary for ChatGPT
"Ejecutá el plan de limpieza de ramas aprobado en BRANCH_HYGIENE_AUDIT_REPORT.md: eliminá las ramas superseded (F06-v1, engine-fix-v1) y la rama de prueba push-test. Creá tags para las ramas de archivo antes de borrarlas."
