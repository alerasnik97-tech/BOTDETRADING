# SINGLE WRITER REMEDIATION REPORT

## 1. Executive Summary
- **Estado**: INCIDENTE EN REMEDIACIÓN.
- **Hallazgo**: Se confirmó que los runners `v50b_limited_rerun_optimized.py` y `v50b_limited_rerun_single_writer_runner.py` carecen de atomicidad en el manejo de locks y no implementan aislamiento de directorios por `run_id`.
- **Acción**: Los resultados de V50B han sido puestos en cuarentena. Se procederá a endurecer el protocolo de bloqueo y aislamiento.

## 2. Incident Classification
- **Tipo**: MULTI-RUNID_CONTAMINATION.
- **Severidad**: P0 (Integridad de Datos).
- **Impacto**: Los rankings de F06, F08 y F12 generados el 14/05/2026 están mezclados con datos de una segunda ejecución no autorizada o concurrente. La auditabilidad es nula.

## 3. Evidence
- **Archivos Afectados**:
    - `trades/V50B_RERUN_TRADES.csv`
    - `audits/V50B_RERUN_REJECTION_AUDIT.csv`
    - `engine_proof/V50B_RERUN_ENGINE_CALL_PROOF.csv`
    - `checkpoints/V50B_RERUN_CHECKPOINTS.csv`
- **Run IDs detectados**: `24bb295d` (Run 1) y `bfe49625` (Run 2).
- **Temporalidad**: Run 1 emitió decisión a las 23:57. Run 2 continuó escribiendo hasta la madrugada.

## 4. Root Cause / Probable Cause
**RACE CONDITION & LACK OF ISOLATION**.
La función `acquire_lock` usa `exists()` seguido de `open(..., 'w')`. En un entorno multihilo o multiproceso, esto permite que dos procesos pasen el check simultáneamente. Además, al escribir directamente en el "official path" mediante `mode='a'`, cualquier proceso con permisos de escritura puede contaminar el dataset maestro.

## 5. Safety Verification
- **test_touched**: NO (Verificado por escaneo de UUIDs vs Fechas).
- **raw_data_mutated**: NO (Parquets intactos).
- **core_drift**: NO (Núcleo intacto).
- **sweep_run**: NO.
- **full_backtest_run**: NO (Solo el rerun afectado).

## 6. Files Changed
| Archivo | Cambio | Motivo |
| :--- | :--- | :--- |
| `src/v6_utils/integrity.py` | [NEW] | Implementación de `AtomicSingleWriter` con `os.O_EXCL`. |
| `scripts/v50b_limited_rerun_optimized.py` | [MODIFY] | Integración de lock atómico y aislamiento de carpetas. |
| `scripts/v50b_limited_rerun_single_writer_runner.py` | [MODIFY] | Integración de lock atómico y aislamiento de carpetas. |
| `tests/remediation/test_integrity.py` | [NEW] | Pruebas unitarias de bloqueo y aislamiento. |

## 7. Single-Writer Protocol (Implemented)
1. **Atomic Lock**: Se utiliza `os.open(path, os.O_CREAT | os.O_EXCL)` para garantizar que solo un proceso tome el control.
2. **Isolation**: Cada corrida escribe sus outputs en `runs/<run_id>/`.
3. **Atomic Publish**: Solo al finalizar con éxito, el runner añade sus resultados a los archivos maestros y genera un `MANIFEST_<run_id>.json`.
4. **Resilience**: El sistema detecta y aborta si el recurso está ocupado, informando el PID y RunID del poseedor actual.

## 8. Tests Run
| Test | Status | Resultado |
| :--- | :--- | :--- |
| `test_atomic_lock_prevention` | PASS | Bloqueo efectivo de segundo escritor. |
| `test_isolation_logic_simulation` | PASS | Verificación de rutas aisladas por RunID. |

## 9. Decision
**REMEDIATED_READY_FOR_CLEAN_SMOKE_RERUN**

## 10. Next Recommended Step
Ejecutar un `preflight_io` con el runner de escritor único para verificar la publicación atómica antes de proceder con una corrida completa de entrenamiento.

## 11. Copy-Paste Summary for ChatGPT
```markdown
## Remediation Status
- **Incident**: MULTI-RUNID_CONTAMINATION Resolved.
- **Protocol**: Atomic Single-Writer + Output Isolation implemented.
- **Tests**: 2/2 PASS.
- **V50B Status**: QUARANTINED (Do not use old rankings).

## Next Step
- Execute `python v50b_limited_rerun_single_writer_runner.py preflight_io`
```
