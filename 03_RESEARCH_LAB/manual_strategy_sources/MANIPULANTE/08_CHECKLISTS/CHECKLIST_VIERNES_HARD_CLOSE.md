# CHECKLIST — VIERNES HARD CLOSE

## Regla Global: VIERNES 16:55 NY = CIERRE OBLIGATORIO

Esta regla es **universal, permanente e irrevocable**.
Aplica a TODAS las cuentas, empresas y modos.

## Checklist Viernes

### Pre-Sesión
- [ ] Confirmar que hoy es viernes
- [ ] Planificar cierre para 16:55 NY como máximo
- [ ] Evaluar si conviene tomar señal considerando margen de cierre

### Durante Sesión
- [ ] Si se abre operación: monitorear hora actual vs 16:55 NY
- [ ] Si operación alcanza TP antes de 16:55 → cerrar normalmente
- [ ] Si operación alcanza SL antes de 16:55 → cerrar normalmente
- [ ] Si BE trigger se activa → mover SL a BE normalmente

### 16:50 NY — Alerta Final
- [ ] **5 minutos para cierre obligatorio**
- [ ] Verificar si hay posición abierta
- [ ] Si hay posición abierta → preparar cierre manual

### 16:55 NY — HARD CLOSE
- [ ] **CERRAR toda posición abierta AHORA**
- [ ] Confirmar ejecución del cierre
- [ ] Anotar resultado como FORCED_FRIDAY_CLOSE
- [ ] Registrar R al momento del cierre

### Post-Cierre
- [ ] Confirmar 0 posiciones abiertas
- [ ] Confirmar 0 órdenes pendientes
- [ ] weekend_holding_violation = false
- [ ] Registrar en ledger con friday_hard_close_executed = true
- [ ] Anotar result_after_global_hard_close_R

## NO HAY OVERRIDE MANUAL
## NO HAY EXCEPCIONES
## APLICA A FTMO, FUNDEDNEXT, PAPER, DEMO, FREE TRIAL, CUALQUIER PROP FIRM
