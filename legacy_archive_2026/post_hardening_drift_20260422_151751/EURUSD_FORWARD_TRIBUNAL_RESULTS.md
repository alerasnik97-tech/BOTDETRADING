# EURUSD Forward Evidence Tribunal - Results

## Auditoría de Juicio Institucional
Se ha certificado el tribunal automático de evidencia forward mediante el script `scratch/run_forward_evidence_tribunal.py`.

## Inputs Auditados
- `results/SCBI_DUAL_LINE_SCOREBOARD.csv`: Fuente primaria de métricas de performance (N, PF, DD).
- `results/SCBI_DUAL_ORCHESTRATOR_STATUS.json`: Fuente de estado de los guards de riesgo.

## Resultado del Rehearsal (Estado Actual)
El tribunal procesó la evidencia disponible al 22 de abril de 2026:
1. **SCBI_M5_GLOBAL**: 
   - N=1, PF=999.0, DD=0.0.
   - **Veredicto**: `PAPER_ONLY (Gathering Sample)`.
   - **Razón**: Muestra insuficiente para cualquier juicio de promoción.
2. **SCBI_CORE**: 
   - N=6, PF=0.45, DD=-1.71.
   - **Veredicto**: `PAPER_ONLY (Gathering Sample)`.
   - **Razón**: Muestra insuficiente. Aunque el PF es bajo, no se dispara bloqueo hasta N=10.

## Capacidades del Tribunal
- **Detección de Checkpoints**: Bloquea promociones hasta N=20 (Demo) y N=40 (Real).
- **Fail-Closed en Riesgo**: Degrada automáticamente a `SUSPENDED` si detecta fallos en la Risk Layer.
- **Detección de Expectativa Negativa**: Bloquea líneas con PF < 1.0 tras el primer hito de muestra (N=10).

## Limitaciones
- El tribunal no lee el "drift vs research" de forma automática todavía (requiere integración con el archivo de baseline). Este juicio permanece en la capa de Review Semanal Manual por ahora.
