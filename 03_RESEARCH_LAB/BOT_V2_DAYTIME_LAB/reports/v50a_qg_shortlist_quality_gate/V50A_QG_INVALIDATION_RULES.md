# V50A-QG INVALIDATION RULES

Cualquier familia seleccionada para V50B serǭ descartada inmediatamente si se cumple uno de estos criterios durante el Gauntlet:

1. **Muestra Insuficiente**: `N_train < 30` o `N_val < 20`.
2. **Pérdida Estructural**: `PF_train < 1.0`. El laboratorio no acepta estrategias perdedoras en IS con la esperanza de que funcionen en OOS.
3. **Falla de Validacin**: `PF_val < 1.10`.
4. **Concentracin Temporal**: Mǭs del 60% de las ganancias concentradas en un slo mes.
5. **Concentracin de Trades**: Los 5 mejores trades explican mǭs del 50% del PnL total.
6. **Leakage**: Deteccin de cualquier trade en 2025+.
7. **Opacidad**: Imposibilidad de auditar un trade especfico mediante logs o visualizacin.
8. **Inconsistencia**: El recalculo de mǸtricas desde el CSV de trades no coincide con el ranking.
