import json
import os
from micro_pilot_gate import config, evaluator

def generate_scorecard_reports(scorecard):
    # 1. Markdown Scorecard
    md = f"""# Micro Pilot Gate Scorecard
## Veredicto: `{scorecard['verdict']}`

**Fecha Evaluación:** `{scorecard['timestamp']}`

---

### 1. Estado de los Gates
- **Research Robustness:** {"✅ PASSED" if scorecard['gates']['research_robustness'] else "❌ FAILED"}
- **Shadow Governance:** {"✅ PASSED" if scorecard['gates']['shadow_governance'] else "❌ FAILED"}
- **Shadow Evidence (N>={config.MIN_SHADOW_TRADES}):** {"✅ PASSED" if scorecard['gates']['shadow_evidence_min'] else "❌ FAILED"}
- **Risk Containment:** {"✅ PASSED" if scorecard['gates']['risk_containment'] else "❌ FAILED"}

### 2. Protocolo de Riesgo Autorizado
- **Riesgo por Trade:** {scorecard['risk_protocol']['risk_per_trade']}
- **Máximo trades/día:** {scorecard['risk_protocol']['max_trades_per_day']}
- **Stop Diario:** {scorecard['risk_protocol']['daily_stop']}
- **Stop Semanal:** {scorecard['risk_protocol']['weekly_stop']}

### 3. Recomendación Institucional
**{scorecard['recommendation']}**

---
*Este gate es un paso intermedio. NO habilita real pleno.*
"""
    with open(config.SCORECARD_MD, 'w', encoding='utf-8') as f:
        f.write(md)

    # 2. Summary TXT
    summary = f"MICRO_PILOT_VERDICT: {scorecard['verdict']}\nGATES_ALL_PASS: {all(scorecard['gates'].values())}\nREC: {scorecard['recommendation']}\n"
    with open(config.SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write(summary)

if __name__ == "__main__":
    gate = evaluator.MicroPilotGate()
    res = gate.evaluate()
    generate_scorecard_reports(res)
    print(f"Reportes de Micro Pilot Gate generados en {config.OUTPUTS_DIR}")
