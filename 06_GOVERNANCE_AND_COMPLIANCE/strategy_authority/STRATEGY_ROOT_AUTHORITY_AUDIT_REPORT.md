# STRATEGY ROOT AUTHORITY AUDIT REPORT

## 1. Status
**STRATEGY_ROOT_AUTHORITY_PARTIAL_MOVES_APPLIED**

## 2. Executive Summary
Se ha realizado una auditoría de autoridad sobre las carpetas de estrategias remanentes en la raíz. Ninguna carpeta presentó evidencia de ser un sistema "Production Certified" o "Active Real/Funding". La mayoría se clasificó como material legacy de investigación o fuentes manuales de hipótesis. Se han aplicado movimientos quirúrgicos hacia `03_RESEARCH_LAB` y `02_INCUBATION_STAGING` para despejar la raíz sin perder material histórico.

## 3. Strategy Root Inventory
| Folder | Status | File Count |
| :--- | :--- | :--- |
| `MANIPULANTE` | MOVED | 171 |
| `ROCKI_AM` | MOVED | 13 |
| `ESTRATEGIAS` | MOVED | 4 |
| `STRATEGIES` | MOVED | 10 |
| `LAB_STRATEGIES` | MOVED | 10 |
| `institutional_research_candidate_lab` | MOVED | 52 |
| `micro_pilot_protocol` | MOVED | 20 |
| `results_REHEARSAL` | MOVED | 14 |
| `shadow_line_lab` | MOVED | 65 |

## 4. Authority Classification
- **MANUAL_STRATEGY_SOURCE:** `MANIPULANTE` (Fuente de hipótesis manuales).
- **INCUBATION_STAGING:** `shadow_line_lab`, `micro_pilot_protocol` (Protocolos de demo/piloto).
- **RESEARCH_LEGACY:** `ROCKI_AM`, `ESTRATEGIAS`, `STRATEGIES`, `LAB_STRATEGIES`, `candidate_lab`, `results_rehearsal`.

## 5. Production / Incubation / Research Decision
- Ningún ítem fue promovido a `01_CORE_PRODUCTION`.
- Ítems de Shadow Trading y Pilot Trading se movieron a `02_INCUBATION_STAGING`.
- El resto se consolidó en `03_RESEARCH_LAB` bajo subcarpetas de "legacy" o "manual".

## 6. ZIP Policy Inside Strategy Folders
- Se realizó un sweep de ZIPs.
- **Resultado:** 0 ZIPs encontrados dentro de las carpetas auditadas.

## 7. Data / Validation / 2025 / 2026 Risk
- Se buscaron términos sensibles. No se detectaron datasets pesados ni scripts activos de backtest de 2025/2026 en estas carpetas. El riesgo de contaminación se considera **BAJO**.

## 8. Move Plan
- [x] MANIPULANTE -> `03_RESEARCH_LAB/manual_strategy_sources/`
- [x] ROCKI_AM -> `03_RESEARCH_LAB/legacy_strategy_sources/`
- [x] ESTRATEGIAS -> `03_RESEARCH_LAB/legacy_strategy_sources/`
- [x] STRATEGIES -> `03_RESEARCH_LAB/legacy_strategy_sources/`
- [x] LAB_STRATEGIES -> `03_RESEARCH_LAB/legacy_strategy_sources/`
- [x] institutional_research_candidate_lab -> `03_RESEARCH_LAB/legacy_strategy_sources/candidate_lab/`
- [x] micro_pilot_protocol -> `02_INCUBATION_STAGING/protocols/micro_pilot/`
- [x] results_REHEARSAL -> `03_RESEARCH_LAB/legacy_strategy_sources/results_rehearsal/`
- [x] shadow_line_lab -> `02_INCUBATION_STAGING/shadow_line/`

## 9. Moves Applied
Todos los listados en el punto 8 han sido ejecutados mediante `git mv`.

## 10. Owner Decisions Required
- Revisión de la nueva estructura de `03_RESEARCH_LAB` para confirmar que la separación legacy/manual es satisfactoria.

## 11. Safety Verification
- strategy_executed: NO
- backtest_run: NO
- raw_data_touched: NO
- validation_touched: NO
- holdout_touched: NO
- 2025_touched: NO
- 2026_touched: NO
- zip_left_active: NO
- production_modified: NO

## 12. Copy-Paste Summary for ChatGPT
`STATUS: STRATEGY_MOVES_APPLIED, ITEMS: 9, DESTINATIONS: 03_RESEARCH & 02_INCUBATION, RISK: LOW, ZIPs: 0.`
