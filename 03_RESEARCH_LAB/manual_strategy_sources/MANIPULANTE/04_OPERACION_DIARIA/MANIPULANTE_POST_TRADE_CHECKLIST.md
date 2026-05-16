# MANIPULANTE — POST-TRADE CHECKLIST

Completar DESPUÉS de cada operación.

## Checklist

- [ ] 1. Resultado registrado en ledger oficial
- [ ] 2. Resultado registrado en ledger observado (si aplica)
- [ ] 3. R result anotado correctamente
- [ ] 4. Status anotado (TP / SL / BE / FORCED_FRIDAY_CLOSE)
- [ ] 5. News Fortress status al momento de entrada registrado
- [ ] 6. Data Quality Mask status registrado
- [ ] 7. Horario de entrada/salida registrado en NY
- [ ] 8. Compliance de max trades/day verificado
- [ ] 9. **VIERNES: Confirmar 0 posiciones abiertas al cierre**
- [ ] 10. **VIERNES: Si se ejecutó hard close → anotar resultado post-close en R**
- [ ] 11. **VIERNES: Verificar weekend_holding_violation = false**
- [ ] 12. Screenshots/evidencia guardada (si aplica)
- [ ] 13. Notas operativas relevantes registradas

## Regla Global Weekend — Post-Trade
Si la operación fue cerrada por hard close viernes:
- Registrar status como `FORCED_FRIDAY_CLOSE`
- Anotar `friday_hard_close_executed = true`
- Anotar `result_after_global_hard_close_R` con el R al momento del cierre
- Confirmar `weekend_holding_violation = false`
