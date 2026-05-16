# Checklist Operativa Diaria - Micro Piloto

> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

Este documento debe ser seguido estrictamente cada día de operación una vez activado el piloto.

## A. Fase Pre-Operativa (Antes de la Sesión)
- [ ] **Check Gate:** Revisar si el gate sigue en `MICRO_PILOT_ALLOWED`.
- [ ] **Shadow Sync:** Confirmar que `shadow_autopilot` dio señal válida hoy.
- [ ] **News Audit:** Revisar calendario económico. ¿Hay noticias rojas (CPI, PPI, NFP, FOMC)? Si sí, evaluar pausa.
- [ ] **Health Check:** ¿Sigue operativo el kill switch?
- [ ] **Emocional:** ¿Estoy en condiciones de seguir el protocolo sin desviarme?

## B. Fase Ejecutiva (Durante la Sesión)
- [ ] **Límites:** Confirmar que el riesgo del trade es <= 0.25%.
- [ ] **Orden:** Ejecutar la orden exacta que indica la lógica (sin overrides).
- [ ] **Monitoreo:** Vigilar la ejecución pero NO intervenir el trade manualmente salvo Kill Switch.
- [ ] **Conteo:** ¿He hecho ya 1 trade hoy? Si sí, cerrar plataforma.

## C. Fase Post-Operativa (Cierre del Día)
- [ ] **Registro:** Anotar resultado (Pips/PnL/Slippage) en el log.
- [ ] **Audit:** Comparar ejecución Real vs ejecución Shadow. ¿Hubo desviación material?
- [ ] **Status Update:** Actualizar `status_template.json` con los nuevos límites de drawdown.
- [ ] **Veredicto Mañana:** ¿El piloto sigue habilitado para mañana? (Basado en límites de riesgo).

## Reglas de Oro Diarias
- 1 solo trade por día.
- Si hay duda, NO hay trade.
- El éxito se mide por la adherencia al protocolo, no por el PnL.
