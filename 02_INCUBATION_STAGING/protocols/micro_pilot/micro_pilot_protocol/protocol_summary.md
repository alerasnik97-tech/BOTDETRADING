# Resumen Ejecutivo: Protocolo de Micro Piloto

## ¿Qué es este protocolo?
Es el conjunto de reglas, límites y checklists que gobernarán la primera fase de trading real controlado (Micro Piloto) una vez que el sistema sea apto.

## ¿Qué NO permite todavía?
**No permite operar en real.** El sistema sigue en **SHADOW_ONLY** debido a que la evidencia acumulada es insuficiente (N < 10).

## ¿Cuándo se podría activar?
Solo cuando el `micro_pilot_gate` emita un veredicto de `MICRO_PILOT_ALLOWED`. Esto requiere completar la fase de incubación Shadow con éxito.

## Parámetros de Riesgo (Resumen)
- **Riesgo/Trade:** 0.10% a 0.25%.
- **Trades/Día:** Máximo 1.
- **Stop Semanal:** 2.5%.
- **Kill Switch:** 5.0% Drawdown.

## Factores de Freno Actuales
- Falta de evidencia shadow (N < 10).
- Gate institucional en estado `NOT_READY`.

## Próximo Paso Crítico
El día que el gate cambie a `ALLOWED`, el primer archivo a consultar es:
`micro_pilot_protocol/activation_checklist.md`

---
**ESTADO DE SEGURIDAD:** 🔒 **BLOQUEADO**
**NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**
