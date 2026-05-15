# OVERNIGHT RUN STATUS - 2026-05-15

## 1. Estado
**COMPLETED_WITH_WARNINGS / BLOCKED**
La corrida V50B Rerun finalizó técnicamente, pero los resultados están bloqueados por una incidencia de integridad (Multi-RunID Contamination). El Smoke Test de Kaggle finalizó con éxito total.

## 2. Evidencia encontrada
| Archivo | Ruta | Timestamp | Tamaño |
| :--- | :--- | :--- | :--- |
| V50B_RERUN_DECISION.md | `.../v50b_limited_real_gauntlet_rerun_sw/` | 2026-05-14 23:57 | 1.8 KB |
| V50B_MULTIRUN_DECISION.md | `.../v50b_rerun_multirun_integrity_incident/` | 2026-05-15 08:12 | 1.2 KB |
| KAGGLE_TRAIN_1M_DECISION.md | `.../results/train_1m_validation/` | 2026-05-15 02:39 | 0.9 KB |
| V50B_RERUN_TRADES.csv | `.../trades/` | 2026-05-15 08:12 | 1.7 MB |
| KAGGLE_TRAIN1M_SAMPLE.csv | `.../results/train_1m_validation/` | 2026-05-15 02:21 | 2.4 MB |

## 3. Procesos vivos
- **PID 18480**: `python.exe ... run-jedi-language-server.py` (StartTime: 14/5/2026 17:23, CPU: 1.6s, RAM: 1.9MB).
- **RUNNERS ACTIVOS**: NINGUNO. Todos los procesos de backtest y validación han finalizado.

## 4. Últimos outputs generados
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_rerun_multirun_integrity_incident/` (Audit de integridad).
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/results/` (Rankings).
- `08_CLOUD_FREE_RUN_LAB/02_KAGGLE_NOTEBOOKS/v50b_kaggle_cloud_smoke_test/results/train_1m_validation/` (Kaggle artifacts).

## 5. Métricas (V50B Rerun)
| Familia | Config | PF_train | PF_val | N_train | N_val | Veredicto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| F06 | F06_RERUN_0001 | 0.65 | 2.18 | 12 | 2 | **REJECTED (Low Train PF)** |
| F08 | - | < 1.0 | - | - | - | **REJECTED (Noise/Throttler)** |
| F12 | - | < 1.0 | - | - | - | **REJECTED (Directional Bias Fix)** |

## 6. Seguridad
- **test_touched**: NO. Escaneo realizado en UUIDs; fechas 2025/2026 inaccesibles.
- **raw_data_mutated**: NO. Verificado mediante auditoría de integridad de parquets.
- **sweep_run**: NO (Solo rerun seguro de familias seleccionadas).
- **optimization_run**: NO.
- **full_backtest_run**: SÍ (V50B Limited Real Gauntlet).

## 7. Diagnóstico
1. **V50B Rerun**: La corrida fue un éxito técnico en cuanto a ejecución pero un fallo estratégico (RED). Además, se detectó una falla en el protocolo de aislamiento de archivos (`Single-Writer`) donde un segundo proceso (`bfe49625`) escribió en los mismos archivos oficiales que el primero (`24bb295d`), invalidando la auditabilidad atómica.
2. **Kaggle**: La infraestructura cloud quedó validada para pruebas TRAIN-only de 1 mes.

## 8. Decisión recomendada
**AUDITAR OUTPUTS Y QUARANTINE**.
1. No usar los resultados de V50B para decisiones de fondeo debido a la contaminación de RunID.
2. Realizar limpieza profunda de la carpeta de resultados.
3. Investigar por qué el sistema de locks permitió la doble escritura.
4. Proceder con el plan de reescritura de F01 (NY Window) recomendado en el veredicto oficial.

## 9. Resumen final para ChatGPT
```markdown
## Overnight Status
- **V50B Rerun**: BLOCKED_MULTI_RUNID_CONTAMINATION (Integrity Incident).
- **Kaggle Smoke Test**: SUCCESS (Infrastructure Ready).
- **Strategic Verdict**: RED (F06, F08, F12 rejected).

## Metrics (V50B)
- F06 PF_train: 0.65 (REJECTED).
- F12: PF < 1.0 (REJECTED).
- Total Trades: 3411 (contaminated by multi-runid).

## Risks
- No test_leakage detected.
- No core_drift detected.
- File isolation protocol failed.

## Next Step
- Quarantine v50b results.
- Audit single_writer_runner.py lock logic.
- Rewrite F01.
```
