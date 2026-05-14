# R1 V49.7 — RESEARCH CHARTER

**Objetivo**: Validar el edge de la estrategia R1 (NY Open Absorption) usando un grid de parámetros reparado y deduplicado, evitando las colisiones de inyección detectadas en V49.

## Hipótesis
La absorción de niveles de liquidez de Asia y Londres durante la apertura de NY genera oportunidades de reversión a la media con un edge explotable, siempre que se usen filtros de wick-to-body y gestión de riesgo estricta (SL en extremos y TP fijos).

## Metodología
- **Universo**: EURUSD.
- **Ventana**: 2020-2024 (TRAIN/VAL).
- **Control de Sesgo**: Deduplicación por hash de trade set.
- **Stress**: Evaluación de degradación ante slippage incremental (0.2, 0.3, 0.5).
- **Selección**: No se usará TEST. Solo candidatos con profit real en TRAIN y estabilidad en VAL serán finalistas.
