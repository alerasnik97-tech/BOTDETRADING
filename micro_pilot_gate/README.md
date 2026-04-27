# Micro Pilot Gate
## Capa de Habilitación para Piloto Real Ultra-Controlado

**Estado:** `RESEARCH_GATE`
**Propósito:** Decidir si una variante shadow puede escalar a un micro-piloto con capital real mínimo.

---

## 1. ¿Qué es el Micro Piloto?
Es un paso intermedio entre la incubación shadow y la producción real. Se opera con micro-lotes y bajo un protocolo de riesgo extremadamente conservador para validar la ejecución técnica sin poner en riesgo el capital principal.

## 2. Requisitos de Habilitación
- **N=10 Shadow:** Al menos 10 trades en el laboratorio shadow con resultados coherentes.
- **Autopilot OK:** La gobernanza shadow debe estar operativa y sin fallos técnicos.
- **Protocolo Aceptado:** Haber validado la checklist de activación y las reglas del kill switch.

## 3. Salidas del Gate
- `micro_pilot_scorecard.md`: Evaluación detallada de los 4 bloques de seguridad.
- `micro_pilot_summary.txt`: Veredicto rápido para automatización.
- `activation_checklist.md`: Guía de pasos para encender el piloto.
- `kill_switch_rules.md`: Reglas duras de detención inmediata.

---
**IMPORTANTE:** Este gate NO significa "Real Ready". Es una validación de seguridad para una fase de prueba limitada. El core productivo del bot permanece aislado de este proceso.
