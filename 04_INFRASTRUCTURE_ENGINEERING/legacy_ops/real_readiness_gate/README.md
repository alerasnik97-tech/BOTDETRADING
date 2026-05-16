# Real Readiness Gate
## Infraestructura Institucional de Decisión para Escalabilidad

**Estado:** V1.0 - Operativo
**Namespace:** `real_readiness_gate/`

---

## 1. ¿Qué es este Gate?
Es un tribunal automatizado y agnóstico que evalúa si una línea estratégica cumple con los estándares institucionales para progresar en su ciclo de vida. 

Actúa como una barrera de seguridad entre:
1. **RESEARCH_ONLY** -> **SHADOW_ONLY** (Incubación aislada)
2. **SHADOW_ONLY** -> **REAL_ELIGIBLE** (Candidato para trading real controlado)

## 2. ¿Qué NO es?
- No es una herramienta de optimización.
- No promueve automáticamente a real.
- No modifica el core productivo ni los runners actuales.

## 3. Funcionamiento
El sistema consume artefactos de investigación y evidencia operativa para validar 4 bloques de robustez:
- **Bloque A (Histórica):** Métricas de backtest (PF, Expectancy, DD, Estabilidad).
- **Bloque B (Operativa):** Disponibilidad de infraestructura y namespaces aislados.
- **Bloque C (Riesgo):** Sensibilidad a noticias y dependencias de timeout.
- **Bloque D (Forward):** Evidencia de ejecución en Shadow Line (N>=20).

## 4. Clasificación de Resultados
- **NOT_READY:** La línea falla en criterios base de backtest u operativos.
- **SHADOW_READY:** Robusta en backtest y riesgo. Lista para iniciar incubación forward.
- **REAL_ELIGIBLE:** Superó todos los gates, incluyendo la validación forward en shadow.

## 5. Cómo Ejecutar
Para evaluar el candidato actual:
```powershell
python real_readiness_gate/evaluator.py
```

Para evaluar una variante específica:
```powershell
python real_readiness_gate/evaluator.py <variant_id>
```

## 6. Salidas (Scorecards)
- `real_readiness_scorecard.json`: Reporte estructurado para telemetría.
- `real_readiness_scorecard.md`: Reporte legible para humanos.
- `real_readiness_summary.txt`: Resumen ejecutivo de veredicto y bloqueo principal.

---
**IMPORTANTE:** Este módulo no tiene "side-effects" sobre la producción. Es una capa puramente analítica y de gobernanza.
