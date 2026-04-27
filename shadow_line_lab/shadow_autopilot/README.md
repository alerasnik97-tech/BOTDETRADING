# Shadow Autopilot
## Coordinación Institucional de la Incubación Shadow

**Estado:** `RESEARCH_ONLY`
**Propósito:** Unificar todos los módulos de la shadow line (Runner, Tribunal, Stack, Checkpoint) en un único pipeline diario automatizado.

---

## 1. Flujo de Ejecución
El Autopilot ejecuta la cadena completa de punta a punta:
1. **Orchestrator:** Ejecuta el runner y genera el ledger inicial.
2. **Tribunal:** Evalúa métricas y emite veredictos de salud.
3. **Daily Stack:** Consolida la bitácora operativa diaria.
4. **Checkpoint Review:** Audita hitos (N=5, 10, 20) y emite recomendaciones de escalado.

## 2. Salidas Principales
- `shadow_line_lab/results/shadow_autopilot_status.json`: Estado técnico unificado.
- `shadow_line_lab/results/shadow_autopilot_log.csv`: Bitácora histórica del autopilot.
- `shadow_line_lab/shadow_autopilot/outputs/shadow_autopilot_summary.md`: Resumen ejecutivo del día.

## 3. Estados del Autopilot
- **SHADOW_AUTOPILOT_OK:** Todo el pipeline corrió exitosamente.
- **SHADOW_AUTOPILOT_WARNING:** El pipeline corrió pero el tribunal o el checkpoint emitieron alertas.
- **SHADOW_AUTOPILOT_BLOCKED_BY_REAL_ERROR:** Fallo técnico en algún módulo del pipeline.

---
**SEGURIDAD:** Este autopilot es autocontenido y no interactúa con los procesos de producción del bot. Su objetivo es puramente el gobierno de la incubación de estrategias en investigación.
