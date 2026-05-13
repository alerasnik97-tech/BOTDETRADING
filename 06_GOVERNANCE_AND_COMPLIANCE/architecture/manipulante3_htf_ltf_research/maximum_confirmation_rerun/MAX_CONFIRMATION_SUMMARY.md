# MAX CONFIRMATION SUMMARY

Estado: MANIPULANTE3_BLOCKED_BY_DATA_OR_ENGINE

## Configs
- cantidad: 5
- ids: CFG_002, CFG_005, CFG_001, CFG_003, CFG_004
- criterio: top 5 por PF_val_net en VALIDATION slippage 0.0; TEST no usado.

## Data/News
- periodo primario: 2020-01 a 2026-04
- AM Fortress v3: fuente primaria
- legacy 2015-2019: no usado
- Tier-1 buffers: standard -1/+5 min; FOMC/tasas -2/+10 min
- rollover: 16:55-17:15 NY bloqueado; forced_exit 16:00 conserva barrera previa
- fail-close: activo via auditorias mensuales PASS y calendario fisico
- news rows: 1106

## Resultados clave
- best config: CFG_002
- PF_val_net slippage 0.2: 0.7893
- PF_test_net slippage 0.2: 0.4120
- N_val: 143
- N_test: 38
- WR_test: 0.3684
- DD_test: -12.4398
- news blocks: 718
- rollover blocks: 3314
- EOM artificial: 76
- verify exacto: True
