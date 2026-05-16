# CHECKLIST — GLOBAL WEEKEND POLICY

## Regla Universal del Sistema Manipulante

**GLOBAL_HARD_CLOSE_BEFORE_MARKET_CLOSE**

| Campo | Valor |
|-------|-------|
| Día | Viernes |
| Hora | 16:55 NY |
| Override manual | NO |
| Aplica a | TODAS las cuentas y modos |
| Origen | Phase32C → Phase32E (promovida a global) |

## Verificaciones Semanales

### Lunes a Jueves
- [ ] Operar normalmente según Phase25
- [ ] No hay restricción adicional por weekend policy

### Viernes — Pre-Sesión
- [ ] Confirmar regla de cierre activa
- [ ] Evaluar margen de cierre antes de tomar señal
- [ ] Si no hay margen → NO OPERAR

### Viernes — 16:55 NY
- [ ] CIERRE OBLIGATORIO de cualquier posición abierta
- [ ] Sin excepciones
- [ ] Sin override manual

### Viernes — Post-Cierre
- [ ] 0 posiciones abiertas confirmado
- [ ] 0 órdenes pendientes confirmado
- [ ] Ledger actualizado con status FORCED_FRIDAY_CLOSE (si aplica)
- [ ] weekend_holding_violation = false confirmado

## Alcance
- ✅ FTMO
- ✅ FundedNext
- ✅ Paper / Demo
- ✅ Free Trial
- ✅ Cualquier evaluación futura
- ✅ Cualquier prop firm futura

## NO cambia
- Estrategia (Phase25)
- TP (1.4R)
- BE (0.4R)
- BF (70%)
- Lógica de entrada
