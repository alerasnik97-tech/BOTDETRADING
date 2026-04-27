import json
import os
import pandas as pd
from shadow_line_lab.checkpoint_review import config

def generate_reports(review):
    # 1. JSON
    with open(config.CHECKPOINT_REVIEW_JSON, 'w', encoding='utf-8') as f:
        json.dump(review, f, indent=2)
    
    # 2. Markdown
    md = f"""# Shadow Checkpoint Review
## Decisión Institucional: `{review['decision']}`

**Hito Objetivo:** N={review['checkpoint_target']} ({review.get('checkpoint_name', 'N/A')})
**Fecha de Revisión:** `{review['timestamp']}`
**Muestra Real (N):** {review['current_n']}

---

### 1. Métricas del Checkpoint
| Métrica | Valor Shadow | Delta vs Baseline |
|---------|--------------|-------------------|
| Profit Factor | {review['metrics'].get('pf', 0)} | {review['baseline_comparison'].get('pf_delta', 0)} |
| Expectancy (R) | {review['metrics'].get('expectancy_r', 0)} | {review['baseline_comparison'].get('expectancy_delta', 0)} |
| Max Drawdown (R) | {review['metrics'].get('max_dd_r', 0)} | - |
| PnL Acumulado (R) | {review['metrics'].get('cumulative_r', 0)} | - |

---

### 2. Análisis Institucional
- **Estado del Hito:** {"ALCANZADO" if review['checkpoint_target'] > 0 else "EN PROGRESO"}
- **Interpretación:** {get_interpretation(review['decision'])}

### 3. Recomendación y Siguiente Paso
**Siguiente Paso:** {get_next_step(review['decision'])}
"""
    with open(config.CHECKPOINT_REVIEW_MD, 'w', encoding='utf-8') as f:
        f.write(md)

    # 3. Escalation Recommendation (SÓLO si es READY)
    if review['decision'] == "READY_FOR_NEXT_GATE":
        generate_escalation_recommendation(review)

def get_interpretation(decision):
    mapping = {
        "CHECKPOINT_NOT_REACHED": "Muestra insuficiente para realizar la revisión del hito.",
        "CONTINUE_INCUBATION": "La evidencia es sana o suficiente para continuar la recolección de datos.",
        "WARNING_REVIEW": "Se detectan desviaciones materiales. Se requiere vigilancia extrema.",
        "HOLD_SHADOW": "Suspensión de la incubación por riesgo estructural o deterioro de performance.",
        "READY_FOR_NEXT_GATE": "Evidencia robusta confirmada. Candidata a revisión de escalado institucional."
    }
    return mapping.get(decision, "N/A")

def get_next_step(decision):
    if decision == "READY_FOR_NEXT_GATE": return "Preparar dossier para tribunal de escalado a Shadow Pilot."
    if decision == "HOLD_SHADOW": return "Detener stack diario y re-evaluar lógica en Research."
    return "Continuar ejecución del Shadow Daily Stack."

def generate_escalation_recommendation(review):
    md = f"""# Shadow Escalation Recommendation
**ESTADO: READY FOR ESCALATION**

La variante ha superado el hito N={review['checkpoint_target']} con métricas satisfactorias.

**Resumen de Evidencia:**
- N Final: {review['current_n']}
- PF Final: {review['metrics']['pf']}
- Expectancy: {review['metrics']['expectancy_r']}R

Se recomienda elevar a revisión de Shadow Pilot.
"""
    with open(config.ESCALATION_RECOMMENDATION_MD, 'w', encoding='utf-8') as f:
        f.write(md)
