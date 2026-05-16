# Checklist de Activación - Micro Piloto

> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

Este checklist DEBE completarse antes de realizar la primera operación en real.

## Estado del Gate Institucional
- [ ] `micro_pilot_gate/outputs/micro_pilot_scorecard.json` muestra `verdict: MICRO_PILOT_ALLOWED`.
- [ ] Evidencia Shadow acumulada >= 10 trades (`shadow_evidence_min: true`).
- [ ] No existen bloqueos pendientes en `ACCIONES_INMEDIATAS.md`.

## Verificación Técnica
- [ ] Conectividad con el broker confirmada y estable.
- [ ] Spread en EURUSD dentro de límites aceptables (< 1.5 pips promedio).
- [ ] Kill switch técnico probado en cuenta demo o con micro-lote de test.
- [ ] Logs de ejecución configurados y guardando en `micro_pilot_protocol/outputs/`.

## Verificación Operativa
- [ ] `risk_limits.md` leído y comprendido.
- [ ] `status_template.json` inicializado correctamente.
- [ ] `daily_operator_checklist.md` impreso o listo para uso.
- [ ] Kill Switch manual identificado y accesible (Un solo click).

## Semáforo de Activación
| Color | Condición | Acción |
|-------|-----------|--------|
| **VERDE** | Todos los puntos anteriores marcados | **ACTIVAR MICRO PILOTO** |
| **AMARILLO** | Gate en ALLOWED pero fallas técnicas menores | **NO ACTIVAR**, corregir primero |
| **ROJO** | Gate en NOT_READY o evidencias < 10 | **PROHIBIDO ACTIVAR** |

## Declaración de Seguridad
La activación del Micro Piloto no es un despliegue masivo. Es una compra de evidencia real. Ante la duda, permanecer en Shadow.

---
**ESTADO ACTUAL:** 🔴 **ROJO** (Pending Gate Approval)
