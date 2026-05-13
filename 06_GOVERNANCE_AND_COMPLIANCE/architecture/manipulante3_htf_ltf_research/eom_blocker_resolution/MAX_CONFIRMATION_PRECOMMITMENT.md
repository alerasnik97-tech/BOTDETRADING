# MAX CONFIRMATION PRECOMMITMENT

Objetivo:
Confirmar fisicamente, sin optimizacion, si el estado RED de MANIPULANTE 3.0 se mantiene al integrar constraints Data/News reales.

Seleccion:
- Fuente unica: MANIPULANTE3_PILOT_RESULTS_VAL.csv.
- Ranking: PF_net de VALIDATION con slippage 0.0.
- TEST no participa en la seleccion.
- Maximo: 5 configs.

Criterios de decision:
- RED si ninguna config supera PF_val_net 1.15 con slippage 0.2 y ninguna supera PF_test_net 1.0 con slippage 0.2, sin blocker de datos/engine y con verify exacto.
- SWEEP_REVIEW_JUSTIFIED solo si una config supera todos los umbrales de VAL, TEST, N, FTMO, EOM y concentracion.
- INCONCLUSIVE si N o cobertura quedan insuficientes sin error duro.
- BLOCKED_BY_DATA_OR_ENGINE si falla Data/News, hay EOM artificial, mismatch de verify o excepcion material.
