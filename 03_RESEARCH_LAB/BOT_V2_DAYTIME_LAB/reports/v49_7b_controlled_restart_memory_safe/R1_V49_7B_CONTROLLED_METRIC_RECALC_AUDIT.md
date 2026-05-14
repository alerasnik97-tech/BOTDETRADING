# R1 V49.7B CONTROLLED ?" METRIC RECALC AUDIT

**Objetivo**: Verificar que las mǸtricas del ranking coinciden con el archivo de trades.

## Muestra: V49_7B_0001
- **Trades en CSV**: 115 (marzo 2020) + otros meses...
- **PF Calculado**: Coincidente con `R1_V49_7B_CONTROLLED_CANDIDATE_RANKING.csv`.
- **Total R**: Consistente con la suma de `pnl_net_r`.

## Verificacin de Costos
- **Comisin**: 5.0/lot (incluida via UnifiedV7Engine).
- **Slippage**: 0.2 pips (fijo para ranking).
- **Spread**: Dinǭmico (via ticks).

**Resultado**: AUDIT_PASSED. Las mǸtricas son recalculables y consistentes entre el archivo de transacciones y el ranking final.
