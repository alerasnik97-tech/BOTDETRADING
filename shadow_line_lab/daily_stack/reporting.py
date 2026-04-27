import json
import os
from shadow_line_lab.daily_stack import config

def generate_daily_scorecard(data):
    # JSON
    with open(config.DAILY_SCORECARD_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    # Markdown
    md = f"""# Shadow Daily Scorecard
## Estado del Stack: `SHADOW_STACK_OK`

**Fecha Proceso:** `{data['run_date']}`
**Fecha Target:** `{data['target_date']}`
**Veredicto Tribunal:** `{data['tribunal_verdict']}`

---

### Resumen de Ejecución
- Shadow Runner Status: `{data['shadow_runner_status']}`
- Signal Found: `{data['shadow_signal_found']}`
- Trade Count: `{data['shadow_trade_count']}`

### Métricas Acumuladas
- Cumulative PnL: `{data['cumulative_R']}R`
- Max Drawdown: `{data['drawdown_R']}R`
- Alertas Activas: {data['alert_count']}

---
*Este reporte es generado por el Shadow Daily Stack aislado.*
"""
    with open(config.DAILY_SCORECARD_MD, 'w', encoding='utf-8') as f:
        f.write(md)

def generate_incubation_summary(summary):
    # JSON
    with open(config.INCUBATION_SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Markdown
    md = f"""# Shadow Incubation Summary
## Estado Global: `{summary['current_tribunal_verdict']}`

**Última Actualización:** `{summary['timestamp']}`

---

### Estadísticas de Incubación
- Total Corridas: {summary['total_shadow_runs']}
- Total Trades: {summary['total_shadow_trades']}
- PnL Acumulado: {summary['cumulative_R']}R
- Max Drawdown: {summary['max_drawdown_R']}R

### Recomendación Institucional
**{summary['recommendation']}**

---
*Aprobado para RESEARCH_ONLY. No para producción.*
"""
    with open(config.INCUBATION_SUMMARY_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    
    # Checkpoints Progress
    generate_progress_checkpoints(summary)

def generate_progress_checkpoints(summary):
    n = summary['total_shadow_trades']
    next_n = 5 if n < 5 else (10 if n < 10 else (20 if n < 20 else 20))
    missing_n = max(0, next_n - n)
    
    template_path = os.path.join(config.OUTPUTS_DIR, "shadow_progress_checkpoints.md_template")
    output_path = os.path.join(config.OUTPUTS_DIR, "shadow_progress_checkpoints.md")
    
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace("{total_shadow_trades}", str(n))
        content = content.replace("{missing_n}", str(missing_n))
        content = content.replace("{cumulative_R}", str(summary['cumulative_R']))
        content = content.replace("{verdict}", summary['current_tribunal_verdict'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
