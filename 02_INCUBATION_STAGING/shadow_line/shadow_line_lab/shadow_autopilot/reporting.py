import json
import os
from shadow_line_lab.shadow_autopilot import config

def generate_final_reports(state):
    # 1. MD Status
    md = f"""# Shadow Autopilot Status
## Estado Global: `{state['overall_status']}`

**Fecha Ejecución:** `{state['run_date']}`
**Fecha Target:** `{state.get('target_date', 'N/A')}`

---

### Pipeline Shadow
| Etapa | Estado |
|-------|--------|
| Runner | `{state['runner_status']}` |
| Tribunal | `{state['tribunal_status']}` |
| Daily Stack | `{state['stack_status']}` |
| Checkpoint | `{state['checkpoint_status']}` |

### Métricas Consolidadas
- Trades Totales: {state.get('trade_count', 0)}
- PnL Acumulado: {state.get('cumulative_R', 0)}R
- Alertas: {state.get('alert_count', 0)}

---
*Pipeline institucional unificado. RESEARCH_ONLY.*
"""
    with open(config.AUTOPILOT_STATUS_MD, 'w', encoding='utf-8') as f:
        f.write(md)

    # 2. Summary JSON
    with open(config.AUTOPILOT_SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    # 3. Summary MD
    summary_md = f"""# Shadow Autopilot Summary
## Veredicto: **{state['overall_status']}**

La incubación shadow del día `{state.get('target_date')}` ha finalizado.

- **Veredicto Tribunal:** `{state['tribunal_status']}`
- **Decisión Checkpoint:** `{state['checkpoint_status']}`
- **Siguiente Paso:** Consultar `shadow_checkpoint_review.md` para detalles del hito actual.

---
"""
    with open(config.AUTOPILOT_SUMMARY_MD, 'w', encoding='utf-8') as f:
        f.write(summary_md)
