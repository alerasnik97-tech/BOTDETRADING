# Shadow Checkpoint Review
## Sistema de Evaluación de Hitos de Incubación

**Estado:** `RESEARCH_ONLY`
**Propósito:** Automatizar la revisión institucional de la Shadow Line en puntos críticos (N=5, 10, 20).

---

## 1. Hitos de Revisión
- **N=5 (Lectura Exploratoria):** Primera verificación de coherencia.
- **N=10 (Primera Señal de Consistencia):** Evaluación de estabilidad media.
- **N=20 (Tribunal de Escalado):** Revisión final de robustez para escalado de gate.

## 2. Decisiones Automáticas
- **CHECKPOINT_NOT_REACHED:** Muestra insuficiente.
- **CONTINUE_INCUBATION:** Evidencia dentro de parámetros esperados.
- **WARNING_REVIEW:** Desviación detectada, requiere vigilancia.
- **HOLD_SHADOW:** Suspensión por riesgo estructural.
- **READY_FOR_NEXT_GATE:** Candidata lista para revisión de escalado institucional.

## 3. Outputs Generados
- `shadow_checkpoint_review.md`: Reporte ejecutivo por hito.
- `shadow_checkpoint_history.csv`: Bitácora histórica de revisiones.
- `shadow_escalation_recommendation.md`: Documento de recomendación (solo si aplica).

---
**SEGURIDAD:** Esta capa es puramente analítica y consultiva. No afecta la ejecución del core ni autoriza trading real sin intervención del comité institucional.
