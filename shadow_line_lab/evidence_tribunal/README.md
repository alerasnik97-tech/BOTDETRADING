# Shadow Evidence Tribunal
## Capa de Gobernanza Forward Aislada

**Estado:** `RESEARCH_ONLY`
**Propósito:** Evaluación automática de la evidencia recolectada en la Shadow Line.

---

## 1. ¿Qué hace esta capa?
El tribunal lee el `shadow_ledger.csv` generado por la shadow line y aplica un conjunto de reglas institucionales (thresholds) para emitir un veredicto sobre la salud y escalabilidad de la estrategia.

## 2. Veredictos Institucionales
- **SHADOW_INCUBATING:** Fase inicial, muestra insuficiente.
- **SHADOW_HEALTHY_EARLY:** Comportamiento positivo con muestra pequeña.
- **SHADOW_WARNING:** Se detectan desviaciones que requieren atención.
- **SHADOW_HOLD:** Pausa obligatoria por riesgo estructural o excesivo drawdown.
- **SHADOW_ESCALATION_CANDIDATE:** Muestra suficiente (N>=20) con métricas que habilitan el gate de escalado.

## 3. Inputs Usados
- `shadow_line_lab/results/shadow_ledger.csv` (Fuente principal de trades).
- `shadow_line_lab/results/shadow_summary.json` (Snapshot operativo).

## 4. Outputs Generados
- `shadow_evidence_scorecard.md`: Reporte legible para humanos.
- `shadow_evidence_scorecard.json`: Data estructurada para telemetría.
- `shadow_alerts.json`: Alertas de seguridad activas.
- `shadow_evidence_log.csv`: Histórico de evaluaciones diarias.

---
**NOTA:** Este tribunal es puramente analítico y opera dentro del namespace de investigación. No tiene capacidad de modificar el core productivo ni autorizar trading real sin revisión humana y gate institucional superior.
