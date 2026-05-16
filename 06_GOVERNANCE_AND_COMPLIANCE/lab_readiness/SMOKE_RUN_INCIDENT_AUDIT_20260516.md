# SMOKE RUN INCIDENT AUDIT 20260516

## 1. Status
**SMOKE_INCIDENT_CONTAINED_LAB_CAN_CONTINUE_AFTER_CLEANUP**

## 2. Executive Summary
Durante la fase de apertura del laboratorio EURUSD Train-Only, se ejecutó un "Smoke Run" que, aunque técnicamente exitoso en demostrar la operatividad del motor, violó varios protocolos de gobernanza institucional al commitear artefactos de investigación pesados a Git y utilizar un flujo de sincronización (`clean-sync-branch`) no autorizado para esta fase. El incidente ha sido contenido mediante la creación de una nueva rama de gobernanza limpia y la cuarentena de los artefactos generados.

## 3. What Happened
- **Rama Utilizada**: Se utilizó `clean-sync-branch` como destino de los cambios, la cual presentaba una divergencia significativa con la rama de gobernanza v3.
- **Mutación del Motor**: Se modificó `research_lab/engine.py` para añadir un fallback de compatibilidad (`generate_signal` -> `signal`) y corregir un error de campo `pair`. Aunque estas correcciones eran necesarias para unificar la interfaz, se realizaron durante la corrida smoke sin una auditoría de código previa separada.
- **Persistencia de Artefactos**: Se ejecutó `git add .`, lo que resultó en el commit `32420260` incluyendo 17 archivos de salida (CSVs, PNGs, JSONs) que suman ~260 KB y añaden ruido al historial de Git.
- **News Policy Conflict**: El reporte `summary.json` indica `news_filter_used: true`, lo cual contradice la política `DEFAULT_NEWS_ENABLED = False` establecida para el pre-lab gate. Investigaciones posteriores indican que este valor fue hardcodeado en `main.py` durante la generación del reporte y no representa un uso real de datos de noticias no auditados.
- **Root Violation**: Se detectó la creación de `000_PARA_CHATGPT.zip` en la raíz del proyecto.

## 4. Commit 32420260 File Audit
| File | Type | Tracked? | Quarantine? | Risk | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `03_RESEARCH_LAB/research_lab/engine.py` | engine/code change | YES | NO | LOW | KEEP (Validated fix) |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/.../*.csv` | output CSV | NO | YES | HIGH | QUARANTINE |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/.../*.png` | output PNG | NO | YES | HIGH | QUARANTINE |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/.../*.json` | summary JSON | NO | YES | MEDIUM | QUARANTINE |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/.../*.md` | smoke report | YES | NO | LOW | MOVE TO GOVERNANCE |

## 5. Clean-sync Impact
La rama `clean-sync-branch` en `origin` está oficialmente contaminada con el commit `32420260`. No debe ser utilizada como base para trabajos futuros hasta que se realice una reconciliación forzada o se retire.

## 6. Output Artifact Audit
Los artefactos locales ubicados en `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/eurusd_train_only_smoke/` contienen métricas de 10 años (2015-2024). No hay evidencia de fuga de datos de 2025/2026 (Leakage). Sin embargo, el volumen de datos (trades.csv, equity_curve.csv) excede lo permitido para el repositorio Git institucional.

## 7. Root Strictness Check
- **Violación**: `000_PARA_CHATGPT.zip` encontrado en raíz.
- **Acción**: Movido a `07_BACKUPS/local_quarantine_do_not_commit/root_cleanup_20260516/`.

## 8. Safety Decision
El entorno es SEGURO para continuar con la fase de **Strategy Research Intake** bajo las siguientes condiciones:
1. No se utilizará la rama `clean-sync-branch`.
2. Se trabajará sobre la rama de gobernanza `governance/smoke-incident-and-strategy-intake-prep-20260516`.
3. Se ignorarán los outputs locales del smoke run en los próximos commits.

## 9. Required Corrective Actions
1. **Quarantine local artifacts**: Mover los reportes locales a una carpeta ignorada por Git.
2. **Standardize engine.py**: Mantener el fix del motor pero documentar el fallback como deuda técnica.
3. **Reset clean-sync**: (Pendiente de aprobación) Realizar un reset de `clean-sync-branch` a la rama de gobernanza actual.

## 10. Copy-Paste Summary for ChatGPT
- **STATUS**: SMOKE_INCIDENT_CONTAINED
- **INCIDENT**: Commit 32420260 pushed outputs and engine mutations to clean-sync-branch.
- **MITIGATION**: Created new governance branch, quarantined root ZIP, audited code changes.
- **SAFE**: YES (Train-only lab remains isolated from 2025/2026).
- **NEXT**: Strategy Research Intake.
