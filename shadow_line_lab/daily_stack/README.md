# Shadow Daily Stack
## Orquestación y Evaluación Unificada de la Shadow Line

**Estado:** `RESEARCH_ONLY`
**Propósito:** Automatizar el flujo diario de ejecución, evaluación y reporte de la incubación shadow.

---

## 1. Componentes del Stack
Este módulo une las piezas individuales de la shadow line en un solo proceso:
1. **Runner Shadow:** Ejecuta la lógica del candidato sobre el dataset del día.
2. **Evidence Tribunal:** Evalúa el ledger resultante y emite veredictos institucionales.
3. **Daily Stack Orchestrator:** Unifica los pasos y genera la bitácora operativa.

## 2. Salidas Principales
- `shadow_line_lab/results/shadow_daily_operational_log.csv`: Bitácora maestra acumulada.
- `shadow_line_lab/daily_stack/outputs/shadow_daily_scorecard.md`: Reporte del día.
- `shadow_line_lab/daily_stack/outputs/shadow_incubation_summary.md`: Resumen acumulado.

## 3. Clasificación de Estados
### Estado del Stack
- **SHADOW_STACK_OK:** Flujo completo exitoso.
- **SHADOW_STACK_WARNING:** Ejecución con alertas no terminales.
- **SHADOW_STACK_BLOCKED_BY_REAL_ERROR:** Fallo técnico en el stack.

### Veredicto del Tribunal (Independiente)
- SHADOW_INCUBATING
- SHADOW_HEALTHY_EARLY
- SHADOW_WARNING
- SHADOW_HOLD
- SHADOW_ESCALATION_CANDIDATE

---
**SEGURIDAD:** Este stack es autocontenido y opera exclusivamente dentro de `shadow_line_lab/`. No tiene interacción con los procesos de producción del bot.
