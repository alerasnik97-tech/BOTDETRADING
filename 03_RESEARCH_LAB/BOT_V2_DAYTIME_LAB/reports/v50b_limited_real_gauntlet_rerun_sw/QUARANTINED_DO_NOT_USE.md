# QUARANTINED - DO NOT USE
**Motivo**: MULTI-RUNID CONTAMINATION
**Fecha de Detección**: 2026-05-15 08:44
**Run IDs Involucrados**: `24bb295d`, `bfe49625`

## Descripción del Incidente
Se detectó que múltiples instancias del runner escribieron de forma secuencial o concurrente en los mismos archivos CSV oficiales:
- `V50B_RERUN_TRADES.csv`
- `V50B_RERUN_REJECTION_AUDIT.csv`
- `V50B_RERUN_ENGINE_CALL_PROOF.csv`
- `V50B_RERUN_CHECKPOINTS.csv`

Esto invalida la integridad de los rankings y la decisión emitida a las 23:57 del 2026-05-14, ya que los datos de la segunda corrida (`bfe49625`) se mezclaron con los de la primera.

## Decisión Institucional
- **NO USAR** los rankings actuales para ninguna promoción de estrategia.
- **ESTADO**: BLOQUEADO.
- **PRÓXIMO PASO**: Hardening del protocolo Single-Writer e implementación de aislamiento por `run_id`.

---
**Auditado por**: Antigravity Senior Integrity Auditor
