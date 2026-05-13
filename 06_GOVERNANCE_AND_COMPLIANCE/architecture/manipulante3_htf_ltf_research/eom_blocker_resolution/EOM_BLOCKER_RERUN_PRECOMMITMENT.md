# EOM BLOCKER RERUN PRECOMMITMENT

Configs congeladas:
- CFG_002
- CFG_005
- CFG_004
- CFG_001
- CFG_003

Seleccion:
- Misma seleccion de Maximum Confirmation.
- No se usa TEST para seleccionar.
- No se agregan variantes.
- No se optimizan parametros.

Periodo:
- TRAIN: 2020-01 a 2021-12
- VAL: 2022-01 a 2022-12
- TEST: 2023-01 a 2026-04

Constraints obligatorios:
- AM Fortress v3.
- Tier-1 buffers.
- Fail-close ante calendario faltante o gaps de calendario.
- Rollover bloqueado 16:55-17:15 NY.
- Bid/ask real.
- FTMO USD 5/lote round-turn.
- Slippage 0.0 y 0.2.
- No EOM artificial en metricas.

Criterios:
- RED solo si no hay artificial EOM en metricas, verify coincide, PF_val_net 0.2 no supera 1.15, PF_test_net 0.2 no supera 1.0 y N es suficiente.
- SWEEP_REVIEW_JUSTIFIED solo si supera criterios precomprometidos de VAL, TEST, N, FTMO, EOM y concentracion.
- INCONCLUSIVE/BLOCKED si N o datos no permiten medicion limpia.
