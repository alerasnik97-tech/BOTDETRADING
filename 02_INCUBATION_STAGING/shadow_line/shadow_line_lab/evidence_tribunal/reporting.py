import json
import os
import pandas as pd
from shadow_line_lab.evidence_tribunal import config

def generate_reports(scorecard):
    # 1. JSON
    with open(config.SCORECARD_JSON, 'w', encoding='utf-8') as f:
        json.dump(scorecard, f, indent=2)
    
    with open(config.ALERTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(scorecard.get("alerts", []), f, indent=2)

    # 2. Markdown
    md = f"""# Shadow Evidence Scorecard
## Veredicto Institucional: `{scorecard['verdict']}`

**Fecha de Evaluación:** `{scorecard['timestamp']}`
**Línea:** `{scorecard.get('variant_id', 'N/A')}`

---

## 1. Métricas de Performance Shadow
| Métrica | Valor |
|---------|-------|
| Total Trades | {scorecard['metrics'].get('total_shadow_trades', 0)} |
| Win Rate | {scorecard['metrics'].get('win_rate', 0)}% |
| Profit Factor | {scorecard['metrics'].get('pf', 0)} |
| Expectancy (R) | {scorecard['metrics'].get('expectancy_R', 0)} |
| Max Drawdown (R) | {scorecard['metrics'].get('max_drawdown_R', 0)} |
| Cumulative PnL (R) | {scorecard['metrics'].get('cumulative_R', 0)} |

---

## 2. Alertas y Observaciones
"""
    if scorecard['alerts']:
        for alert in scorecard['alerts']:
            md += f"### [{alert['severity']}] {alert['code']}\n- **Explicación:** {alert['explanation']}\n- **Acción Recom:** {alert['recommended_action']}\n\n"
    else:
        md += "No se detectaron alertas materiales.\n"

    md += """
---
## 3. Interpretación Institucional
- **SHADOW_INCUBATING:** Muestra insuficiente para conclusiones.
- **SHADOW_HEALTHY_EARLY:** Comportamiento inicial positivo.
- **SHADOW_WARNING:** Desviación detectada, requiere vigilancia.
- **SHADOW_HOLD:** Suspensión por riesgo estructural o técnico.
- **SHADOW_ESCALATION_CANDIDATE:** Listo para revisión de gate superior.
"""
    with open(config.SCORECARD_MD, 'w', encoding='utf-8') as f:
        f.write(md)

    # 3. Summary TXT
    summary = f"VERDICT: {scorecard['verdict']}\nTRADES: {scorecard['metrics'].get('total_shadow_trades', 0)}\nPNL_R: {scorecard['metrics'].get('cumulative_R', 0)}\nALERTS: {len(scorecard['alerts'])}\n"
    with open(config.SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write(summary)

    # 4. Evidence Log (CSV)
    log_entry = {
        "timestamp": scorecard["timestamp"],
        "verdict": scorecard["verdict"],
        "trades": scorecard["metrics"].get("total_shadow_trades", 0),
        "cumulative_R": scorecard["metrics"].get("cumulative_R", 0),
        "pf": scorecard["metrics"].get("pf", 0),
        "max_dd": scorecard["metrics"].get("max_drawdown_R", 0)
    }
    df_log = pd.DataFrame([log_entry])
    header = not os.path.exists(config.EVIDENCE_LOG)
    df_log.to_csv(config.EVIDENCE_LOG, mode='a', index=False, header=header)
